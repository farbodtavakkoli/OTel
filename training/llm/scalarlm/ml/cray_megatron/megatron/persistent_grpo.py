import json
import logging
import math
import os
import time

import torch
import yaml

from cray_infra.training.training_job_status import TrainingJobStatus
from cray_infra.util.get_job_config import get_job_config, get_job_config_path

from cray_megatron.collectives.main_rank_only import is_main_rank
from cray_megatron.megatron import stop_flag
from cray_megatron.megatron.dataset.data_loader import DataLoader
from cray_megatron.megatron.dataset.load_grpo_dataset import set_dataset_path_override
from cray_megatron.megatron.training_loop import (
    build_adapter_metadata,
    filter_checkpoint,
    get_optimizer,
    get_scheduler,
)

logger = logging.getLogger(__name__)

POLL_S = 0.2
HEARTBEAT_S = 60
IDLE_TIMEOUT_S = 1800


def is_persistent_grpo(job_config):
    grpo = job_config.get("grpo") or {}
    return job_config.get("training_mode") == "grpo" and bool(grpo.get("persistent"))


def _write_json_atomic(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


class PersistentGrpo:
    def __init__(self, loop):
        self.loop = loop
        self.job_dir = os.path.abspath(get_job_config()["job_directory"])
        self.jobs_root = os.path.dirname(self.job_dir)
        self.inbox = os.path.join(self.job_dir, "inbox")
        self.outbox = os.path.join(self.job_dir, "outbox")
        self.global_step = 0

    def run(self):
        state = self.loop.training_state
        state.start_time = time.time()
        state.model_info.pop("init_adapter_checkpoint", None)
        state.model_info["model"].train()

        if os.path.isdir(self.outbox) and os.listdir(self.outbox):
            raise RuntimeError(
                f"persistent GRPO session {self.job_dir} already served requests; "
                "a restarted session has lost its adapter state, start a new session"
            )
        if is_main_rank():
            for d in (self.inbox, self.outbox):
                os.makedirs(d, exist_ok=True)
                os.chmod(d, 0o777)
            self._touch_status({"persistent_cycles": 0})
        logger.info(f"persistent GRPO: model loaded, waiting for requests in {self.inbox}")

        seq = 1
        while True:
            request = self._wait_for_request(seq)
            stop = request is None or bool(request.get("stop"))
            if self.loop._stop_requested_on_any_rank(stop):
                break
            result = self._run_cycle(seq, request)
            if is_main_rank():
                _write_json_atomic(os.path.join(self.outbox, f"{seq:06d}.json"), result)
                self._touch_status({"persistent_cycles": seq})
            seq += 1

        logger.info(f"persistent GRPO: session ends after {seq - 1} request(s)")

    def _wait_for_request(self, seq):
        path = os.path.join(self.inbox, f"{seq:06d}.json")
        t0 = last_heartbeat = time.time()
        while True:
            if os.path.exists(path):
                with open(path) as f:
                    return json.load(f)
            if stop_flag.was_stop_requested():
                logger.info("persistent GRPO: stop requested by signal")
                return None
            now = time.time()
            if now - t0 > IDLE_TIMEOUT_S:
                logger.warning(f"persistent GRPO: no request {seq} within {IDLE_TIMEOUT_S}s, stopping")
                return None
            if is_main_rank() and now - last_heartbeat > HEARTBEAT_S:
                self._touch_status({})
                last_heartbeat = now
            time.sleep(POLL_S)

    def _run_cycle(self, seq, request):
        state = self.loop.training_state
        model = state.model_info["model"]
        steps = int(request.get("steps", 1))
        ga = int(request["gradient_accumulation_steps"])
        t_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        set_dataset_path_override(os.path.join(self.inbox, request["data"]))
        state.optimizer = get_optimizer(model)
        state.scheduler = get_scheduler(state.optimizer, steps)
        loader = iter(
            DataLoader(model=model, tokenizer=state.model_info["tokenizer"], starting_epoch=0)
        )

        losses, step_times, grpo_steps, micro_times = [], [], [], []
        for _ in range(steps):
            state.current_step = self.global_step
            t_step = time.time()
            state.optimizer.zero_grad()
            accumulated = None
            for accum_step in range(ga):
                loss = self.loop.training_step_accumulate(next(loader), accum_step, ga)
                accumulated = loss if accumulated is None else accumulated + loss
            model.backward_sync()
            stepped = self.loop.optimizer_step()
            _, avg = self.loop.sync_loss(accumulated / ga)
            if not stepped:
                state.nan_steps += 1
                avg = float("nan")
            stats = self.loop._grpo_step_stats
            micro_times.extend(x["time_s"] for x in stats)
            grpo_steps.append(self._summarize(stats))
            step_time = time.time() - t_step
            self.loop.print_training_step_info(avg, step_time)
            self.loop._log_grpo_step()
            losses.append(avg)
            step_times.append(step_time)
            self.global_step += 1
        state.optimizer.zero_grad(set_to_none=True)
        train_s = time.time() - t_start

        t_save = time.time()
        self._publish(request["model_name"], losses)
        save_s = time.time() - t_save

        gib = 1024 ** 3
        peak = torch.cuda.max_memory_allocated() / gib if torch.cuda.is_available() else None
        if is_main_rank():
            logger.info(
                f"persistent GRPO request {seq}: published {request['model_name']} "
                f"train {train_s:.2f}s save {save_s:.2f}s peak {peak} GiB"
            )
        return {
            "seq": seq,
            "model_name": request["model_name"],
            "losses": losses,
            "step_times_s": step_times,
            "microbatch_times_s": micro_times,
            "grpo_steps": grpo_steps,
            "train_s": train_s,
            "save_s": save_s,
            "peak_mem_gib": round(peak, 2) if peak is not None else None,
        }

    @staticmethod
    def _summarize(stats):
        if not stats:
            return {}
        tokens = sum(x["tokens"] for x in stats)
        w = [x["tokens"] / max(tokens, 1) for x in stats]
        return {
            "loss": sum(x["loss"] for x in stats) / len(stats),
            "mean_ratio": sum(wi * x["mean_ratio"] for wi, x in zip(w, stats)),
            "clip_frac": sum(wi * x["clip_frac"] for wi, x in zip(w, stats)),
            "tokens": tokens,
            "sequences": sum(x["sequences"] for x in stats),
            "micro_batches": len(stats),
        }

    def _publish(self, model_name, losses):
        model = self.loop.training_state.model_info["model"]
        if hasattr(model, "unwrap_model"):
            state_dict = model.unwrap_model()
        else:
            state_dict = filter_checkpoint(model.model, model.model.state_dict())
        if not is_main_rank():
            return

        out_dir = os.path.join(self.jobs_root, model_name)
        os.makedirs(out_dir, exist_ok=True)
        checkpoint = {
            "model_state_dict": state_dict,
            "metadata": build_adapter_metadata(),
            "step": self.global_step - 1,
            "epoch": 0,
        }
        tmp = os.path.join(out_dir, "checkpoint_0.pt.tmp")
        torch.save(checkpoint, tmp)
        os.replace(tmp, os.path.join(out_dir, "checkpoint_0.pt"))

        with open(get_job_config_path()) as f:
            config = yaml.safe_load(f) or {}
        config["job_directory"] = out_dir
        with open(os.path.join(out_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(config, f)

        now = time.time()
        first = self.global_step - len(losses)
        history = [
            {"step": first + i, "loss": None if math.isnan(l) else l, "epoch": 0, "time": now - self.loop.training_state.start_time}
            for i, l in enumerate(losses)
        ]
        _write_json_atomic(
            os.path.join(out_dir, "status.json"),
            {
                "status": TrainingJobStatus.COMPLETED,
                "start_time": now,
                "last_updated": now,
                "history": history,
                "model_name": model_name,
                "persistent_session": os.path.basename(self.job_dir),
            },
        )

    def _touch_status(self, extra):
        path = os.path.join(self.job_dir, "status.json")
        try:
            with open(path) as f:
                status = json.load(f)
        except Exception:
            status = {}
        status.update(extra)
        status["status"] = TrainingJobStatus.TRAINING
        status["last_updated"] = time.time()
        _write_json_atomic(path, status)
