from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field, replace

from torchtitan.experiments.rl.environment import TokenEnv
from torchtitan.experiments.rl.rollout.advantage import AdvantageEstimator
from torchtitan.experiments.rl.rollout.rollouter import Rollouter, RolloutWorker
from torchtitan.experiments.rl.rollout.types import Rollout, RolloutGroup
from torchtitan.experiments.rl.rubrics import RewardFn, Rubric, RubricOutput

from titanrl_telecom.data import TelecomDataset, TelecomSample
from titanrl_telecom.env import TelecomEnv

_COMMON_DIR = os.environ.get("GRPO_COMMON_DIR") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "grpo_common")
)
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)
import judge_reward  # noqa: E402  (grpo_common/judge_reward.py)


class RewardJudgePlaceholder(RewardFn):

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: object) -> float:
        raise RuntimeError("TelecomWorker.score_group must be used")


def _completion_text(rollout: Rollout) -> str:
    if not rollout.turns:
        return ""
    msg = rollout.turns[-1].completion_message
    return (msg.get("content") or "") if msg else ""


class TelecomWorker(RolloutWorker):
    @dataclass(kw_only=True, slots=True)
    class Config(RolloutWorker.Config):
        rubric: Rubric.Config = field(
            default_factory=lambda: Rubric.Config(
                reward_fns=[RewardJudgePlaceholder.Config(weight=1.0)]
            )
        )
        message_env: TelecomEnv.Config = field(default_factory=TelecomEnv.Config)
        token_env: TokenEnv.Config = field(
            default_factory=lambda: TokenEnv.Config(
                max_rollout_tokens=6144, max_num_turns=1
            )
        )
        advantage: AdvantageEstimator.Config = field(
            default_factory=lambda: AdvantageEstimator.Config(should_std_normalize=True)
        )
        trace_dir: str = os.environ.get("TITAN_TELECOM_TRACE_DIR", "")  # "" = no trace

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._trace_path = None
        if config.trace_dir:
            os.makedirs(config.trace_dir, exist_ok=True)
            self._trace_path = os.path.join(
                config.trace_dir, f"rollout_trace.{os.getpid()}.jsonl"
            )
        self._gen_windows: dict[int, tuple[float, float]] = {}

    async def score_group(
        self, rollouts: list[Rollout], env_input: TelecomSample
    ) -> list[RubricOutput]:
        comps = [_completion_text(r) for r in rollouts]
        n = len(rollouts)
        finish = ["length" if r.status.is_truncated() else "stop" for r in rollouts]
        t0 = time.time()
        out = await asyncio.to_thread(
            judge_reward.score_batch,
            [env_input.question] * n,
            comps,
            [env_input.reference] * n,
            [env_input.source] * n,
            finish,
        )
        t1 = time.time()
        gid = rollouts[0].group_id if rollouts else -1
        gen = self._gen_windows.pop(gid, (None, None))
        pv = None
        for r in rollouts:
            if r.turns and r.turns[0].max_policy_version is not None:
                pv = r.turns[0].max_policy_version
                break
        if self._trace_path:
            rec = {
                "group_id": gid,
                "sample_id": env_input.id,
                "source": env_input.source,
                "policy_version": pv,
                "t_gen_start": gen[0],
                "t_gen_end": gen[1],
                "t_judge_start": t0,
                "t_judge_end": t1,
                "judge_timing_s": out["timing_s"],
                "completion_tokens": [
                    len(r.turns[-1].completion_token_ids) if r.turns else 0
                    for r in rollouts
                ],
                "status": [r.status.value for r in rollouts],
                "rewards": out["reward"],
                "judge": out["judge"],
            }
            with open(self._trace_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return [
            RubricOutput(
                reward=float(rw),
                reward_breakdown={"judge": (j if j is not None else -1.0)},
            )
            for rw, j in zip(out["reward"], out["judge"], strict=True)
        ]

    async def run_group(self, *, generate_fn, sample, group_id, group_size, sampling):
        envs = self.make_env_group(sample=sample, group_size=group_size)
        t_gen0 = time.time()
        try:
            rollouts = await asyncio.gather(
                *(
                    self._run_single_rollout(
                        generate_fn=generate_fn,
                        env=env,
                        sampling=(
                            sampling
                            if sampling.seed is None
                            else replace(sampling, seed=sampling.seed + sample_idx)
                        ),
                        group_id=group_id,
                        rollout_id=sample_idx,
                    )
                    for sample_idx, env in enumerate(envs)
                )
            )
        finally:
            await asyncio.gather(*(env.close() for env in envs), return_exceptions=True)
        self._gen_windows[group_id] = (t_gen0, time.time())

        outputs = await self.score_group(rollouts, sample)
        for rollout, output in zip(rollouts, outputs, strict=True):
            rollout.reward = output.reward
            rollout.reward_breakdown = output.reward_breakdown

        group = RolloutGroup(group_id=group_id, rollouts=rollouts)
        advantages = self.advantage_estimator(group)
        for rollout, advantage in zip(group.rollouts, advantages, strict=True):
            rollout.advantage = advantage
        return group


class TelecomRollouter(Rollouter):
    @dataclass(kw_only=True, slots=True)
    class Config(Rollouter.Config):
        train_dataset: TelecomDataset.Config = field(default_factory=TelecomDataset.Config)
        validation_dataset: TelecomDataset.Config = field(
            default_factory=lambda: TelecomDataset.Config(shuffle=False, limit=8)
        )
        worker: TelecomWorker.Config = field(default_factory=TelecomWorker.Config)
