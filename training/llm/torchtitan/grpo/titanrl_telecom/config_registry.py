from __future__ import annotations

import json
import os

from renderers import Qwen3RendererConfig

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.config import (
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.rl.actors.generator import (
    SamplingConfig,
    VLLMCudagraphConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.experiments.rl.environment import TokenEnv
from torchtitan.experiments.rl.losses import GRPOLoss
from torchtitan.experiments.rl.models.cast_linear import LMHeadCastConverter
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RenderersLibraryConfig
from torchtitan.experiments.rl.rollout.advantage import AdvantageEstimator
from torchtitan.experiments.rl.routing.inter_generator_router import (
    InterGeneratorRouter,
)
from torchtitan.experiments.rl.routing.strategies import LeastLoadedRoutingStrategy
from torchtitan.models.qwen3 import model_registry

from titanrl_telecom.data import DEFAULT_PROMPTS, TelecomDataset
from titanrl_telecom.worker import TelecomRollouter, TelecomWorker

# The `rl_telecom_qwen3_*` presets below take their paths from the environment; `rl_telecom` gets
# everything from $TITAN_GRPO_CONFIG instead and ignores these.
_SHARED_DIR = os.environ.get("GRPO_SHARED_DIR", "./shared")
MODELS_DIR = os.environ.get("TITAN_MODELS_DIR", os.path.join(_SHARED_DIR, "models"))
DUMP_ROOT = os.environ.get("TITAN_DUMP_ROOT", os.path.join(_SHARED_DIR, "runs", "torchtitan_rl"))
RUN_NAME = os.environ.get("TITAN_RUN_NAME", "run")

CONSOLE_KEYS = [
    "timing/",
    "perf/",
    "loss/",
    "rollout_reward/",
    "train/response_length",
    "trainer/",
    "generator/",
]


def _telecom_config(
    *,
    flavor: str,
    hf_dir: str,
    num_training_steps: int,
    prompts_per_step: int,
    group_size: int,
    max_response_tokens: int,
    max_prompt_tokens: int,
    num_generators: int,
    trainer_dp_shard: int,
    trainer_tp: int,
    lr: float,
    save_final: bool,
    generator_tp: int = 1,
    compile_enable: bool = True,
    data: str = DEFAULT_PROMPTS,
    temperature: float = 1.0,
    seed: int = 42,
    dump_folder: str | None = None,
    attn_backend: str | None = None,
) -> Controller.Config:
    max_total_tokens = max_prompt_tokens + max_response_tokens
    dump_folder = dump_folder or os.path.join(DUMP_ROOT, RUN_NAME)
    trace_dir = os.environ.get("TITAN_TELECOM_TRACE_DIR") or os.path.join(
        dump_folder, "rollout_trace"
    )
    # The "varlen" default routes the generator to vLLM's CUSTOM backend, which needs
    # FlashAttention-3 on CUDA SM 9.0 and fails there. Set TITAN_ATTN_BACKEND=flex on SM 9.0; see
    # training/llm/torchtitan/grpo/README.md.
    attn_backend = attn_backend or os.environ.get("TITAN_ATTN_BACKEND", "varlen")
    return Controller.Config(
        model_spec=model_registry(
            flavor,
            seq_len=max_total_tokens,
            attn_backend=attn_backend,
            converters=[LMHeadCastConverter.Config()],  # fp32 logits, bf16 forward
        ),
        hf_assets_path=hf_dir,
        dump_folder=dump_folder,
        async_loop=AsyncLoopConfig(
            num_training_steps=num_training_steps,
            num_prompts_per_train_step=prompts_per_step,
            num_samples_per_prompt=group_size,
            target_offpolicy_steps=0,  # synchronous: on-policy GRPO loop
            window_fraction=None,  # strict FIFO
            validation=ValidationConfig(num_samples=0),
        ),
        compile=CompileConfig(enable=compile_enable, backend="aot_eager"),
        rollouter=TelecomRollouter.Config(
            train_dataset=TelecomDataset.Config(path=data, seed=seed, shuffle=True),
            validation_dataset=TelecomDataset.Config(path=data, shuffle=False, limit=8),
            worker=TelecomWorker.Config(
                token_env=TokenEnv.Config(
                    max_rollout_tokens=max_total_tokens, max_num_turns=1
                ),
                advantage=AdvantageEstimator.Config(should_std_normalize=True),
                trace_dir=trace_dir,
            ),
            worker_pool_size=4,
        ),
        renderer=RenderersLibraryConfig(
            renderers_config=Qwen3RendererConfig(enable_thinking=False)
        ),
        num_generators=num_generators,
        generator_router=InterGeneratorRouter.Config(
            strategy=LeastLoadedRoutingStrategy.Config()
        ),
        metrics=MetricsProcessor.Config(
            enable_wandb=False,
            enable_tensorboard=True,
            console_log_keys_train=CONSOLE_KEYS,
        ),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=lr, betas=(0.9, 0.98), weight_decay=0.1),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=0, min_lr_factor=1.0),
            training=TrainingConfig(
                num_tokens_per_microbatch_per_dp_rank=max_total_tokens,
                max_context_length=max_total_tokens,
            ),
            debug=DebugConfig(seed=seed),
            parallelism=ParallelismConfig(
                data_parallel_replicate_degree=1,
                data_parallel_shard_degree=trainer_dp_shard,
                tensor_parallel_degree=trainer_tp,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10_000,  # no intermediate DCP saves (they are the full model)
                last_save_model_only=True,
                last_save_in_hf=save_final,
                keep_latest_k=2,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(clip_eps=0.2),  # no KL term: beta = 0
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1, tensor_parallel_degree=generator_tp
            ),
            cudagraph=VLLMCudagraphConfig(enable=True),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=temperature, top_p=1.0, max_tokens=max_response_tokens, seed=seed
            ),
        ),
    )


def rl_telecom() -> Controller.Config:
    return _telecom_config(**json.loads(os.environ["TITAN_GRPO_CONFIG"]))


_Q4B = os.path.join(MODELS_DIR, "Qwen--Qwen3-4B")
_Q32B = os.path.join(MODELS_DIR, "Qwen--Qwen3-32B")


def rl_telecom_qwen3_4b_smoke() -> Controller.Config:
    return _telecom_config(
        flavor="4B", hf_dir=_Q4B, num_training_steps=2, prompts_per_step=8, group_size=4,
        max_response_tokens=256, max_prompt_tokens=4096, num_generators=1,
        trainer_dp_shard=1, trainer_tp=1, lr=1e-6, save_final=True,
    )


def rl_telecom_qwen3_4b_bench_2gpu() -> Controller.Config:
    return _telecom_config(
        flavor="4B", hf_dir=_Q4B, num_training_steps=22, prompts_per_step=32, group_size=8,
        max_response_tokens=1024, max_prompt_tokens=4096, num_generators=1,
        trainer_dp_shard=1, trainer_tp=1, lr=1e-6, save_final=False,
    )


def rl_telecom_qwen3_4b_bench_4gpu() -> Controller.Config:
    return _telecom_config(
        flavor="4B", hf_dir=_Q4B, num_training_steps=22, prompts_per_step=32, group_size=8,
        max_response_tokens=1024, max_prompt_tokens=4096, num_generators=3,
        trainer_dp_shard=1, trainer_tp=1, lr=1e-6, save_final=False,
    )


def rl_telecom_qwen3_32b_smoke() -> Controller.Config:
    return _telecom_config(
        flavor="32B", hf_dir=_Q32B, num_training_steps=2, prompts_per_step=8, group_size=4,
        max_response_tokens=256, max_prompt_tokens=4096, num_generators=1,
        trainer_dp_shard=3, trainer_tp=1, lr=1e-6, save_final=False,
    )


def rl_telecom_qwen3_32b_bench_4gpu() -> Controller.Config:
    return _telecom_config(
        flavor="32B", hf_dir=_Q32B, num_training_steps=22, prompts_per_step=32, group_size=8,
        max_response_tokens=1024, max_prompt_tokens=4096, num_generators=1,
        trainer_dp_shard=3, trainer_tp=1, lr=1e-6, save_final=False,
    )
