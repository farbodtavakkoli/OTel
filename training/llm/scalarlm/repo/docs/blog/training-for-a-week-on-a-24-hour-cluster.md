# Training for a Week on a 24-Hour Cluster

*2026-05-20*

A user pinged us recently with a question that turned into a useful design conversation. Her training run was projected to take 3.5 days. The cluster's SLURM scheduler had a 24-hour per-job time limit. She'd set `steps_per_checkpoint = 1000`. The slurm timeout fired at around step 2800, and the job restarted from the last checkpoint at step 2000 — losing ~800 steps, roughly a third of a day's compute, every 24 hours.

The fix is interesting because almost none of it is in code you'd write yourself. It's mostly in how a few pieces are arranged.

## Why per-job time caps are a feature, not a bug

It's tempting to just raise the cluster's max job time to a week. Don't. A per-job cap is doing real work:

- **It verifies your checkpoints actually work.** If your training script never gets restarted, you don't find out that the checkpoint loader is broken, or that the resumed-from-disk optimizer state has a subtle dtype mismatch, until something else forces a restart — and at that point you've lost a week. Forced periodic restarts surface the bug on day one.
- **It surfaces hardware and storage faults early.** A node that's about to fail tends to fail somewhere — the more checkpoint-and-restart cycles you do, the more likely you catch the failure with intact state.
- **It load-balances.** Two researchers can have week-long jobs running concurrently; they take turns occupying the cluster instead of one of them getting blocked for a week.

So the long-job problem becomes: how do you make 24-hour caps invisible to the user, while keeping the discipline they provide?

## What the user writes

```python
import scalarlm
llm = scalarlm.SupermassiveIntelligence()
llm.train(
    data=...,
    train_args={
        "max_steps": 30_000,
        "timeout": 3.5 * 86400,   # 3.5 days, total
        "steps_per_checkpoint": 100,
    },
)
```

`timeout` is the user's **total** wall-clock budget. Not per-slice. The user does not need to know how many SLURM slices their job will become.

## What happens on the server

ScalarLM takes that single submission and arranges it into a chain of SLURM jobs. Each is ≤ `max_train_time` (the cluster's hard cap, 24h by default). The user sees one job; SLURM sees a sequence.

Three pieces make this work:

### 1. A grace-window signal at the slurm boundary

`launch_training_job.py` adds `--signal=B:TERM@300` to every sbatch invocation. SLURM sends SIGTERM to the batch shell 5 minutes before the slice's `--time` runs out (and SIGKILLs at the deadline if you ignore it).

The bash entrypoint is a single line: `exec mpirun --allow-run-as-root python main.py`. Because of `exec`, the slurm batch shell's PID *becomes* mpirun's PID, so slurm's SIGTERM lands on mpirun directly — no bash trap to write, no PID juggling. mpirun's standard signal forwarding takes it from there to each rank's python process.

### 2. A signal handler that checkpoints, then exits

The trainer's signal handler does almost nothing: sets a module-level flag and returns. It does *not* call `sys.exit`. The training loop polls the flag at every step boundary:

```python
if stop_flag.was_stop_requested():
    self.training_state.should_stop_training = True
if self.training_state.should_stop_training:
    break
```

After the loop breaks, `TrainingLoop.train()` runs `self.checkpoint()` — the same final-checkpoint call that fires when `max_steps` is reached. This is the line that fixed the user's problem: she was losing 800 steps because slurm SIGKILLed mid-loop and the last checkpoint was 800 steps stale. With the SIGTERM handler in place, the trainer always writes a fresh checkpoint at the moment the slice ends. She loses at most one step.

The grace window default (5 minutes) is generous on purpose — enough room for the in-flight step to finish *and* the checkpoint write to complete on most mid-sized models. If your model is large enough that checkpoint writes routinely exceed 5 minutes, you bump `signal_grace_seconds` in `cray-config.yaml`. If the checkpoint *does* exceed the grace window, the previous checkpoint is still safely on disk — the next slice resumes from it, losing only the work done after that checkpoint. Same failure mode as before; never worse.

### 3. A reconciler that doesn't know about long jobs

Here's the part where ScalarLM intentionally does *not* add new infrastructure.

The control plane already runs a periodic task called `restart_megatron_jobs`. Every 30 seconds it diffs status.json files against `squeue` and resubmits anything in `TRAINING`/`QUEUED` that's missing from the slurm queue. It was originally there for pod restarts and node preemption. We discovered while building this feature that it covers slurm slice timeouts for free, with one small condition: the trainer must *not* mark the job as `COMPLETED` when the slice was cut short by a signal.

That's three lines in `megatron_trainer.py`:

```python
TrainingLoop(self.training_harness).train()
if not stop_flag.was_stop_requested():
    self.training_harness.update_status(status=TrainingJobStatus.COMPLETED)
```

When SIGTERM cut the slice short, status stays at `TRAINING`. The reconciler sees a `TRAINING` job missing from `squeue` and resubmits via `start_slurm_job(config)`. The same code path that handles a node failure handles a slice boundary. There is no separate "auto-relaunch on timeout" subsystem — there's the trainer doing its job, and the reconciler doing the job it already did, tied together by a single status-discipline rule.

When the loop ends naturally (max_steps reached, or the user's total `timeout` was exhausted by the `TimeoutCallback`), no signal was received, status flips to `COMPLETED`, and the reconciler correctly leaves it alone.

### 4. Cross-slice budget tracking

The user's total `timeout` only makes sense if elapsed time carries across slices. After every slice, the trainer persists `accumulated_train_seconds` to status.json. The next slice loads it back and `TimeoutCallback` compares (this slice's elapsed + carried-forward elapsed) against the user's total budget. The 3.5-day job stops after 3.5 days of *actual training*, regardless of how many slurm slices it took.

## What gets stored where

```
{job_dir}/
├── status.json              # status, job_id, history, accumulated_train_seconds
├── config.yaml              # the user's train_args + server-derived fields
├── checkpoint_{step}.pt     # rotated, last N kept
├── slurm-{id}.out           # one per slice — useful for debugging
└── ...
```

Everything that needs to survive a slice boundary lives in status.json or on disk in the job directory. There are no sentinel files, no per-job state in the API server, no callbacks-via-RPC. The job directory is content-addressed by the user's `train_args` + dataset hash, so re-submissions land in the same place and resume picks up from the latest checkpoint without any explicit `--resume` flag.

## What this looks like in practice

For the user's 3.5-day job on a 24-hour cluster:

| | Before | After |
|---|---|---|
| Slurm hits 24h | SIGKILL | SIGTERM 5 min before, SIGKILL at the deadline if ignored |
| Last checkpoint when slurm fires | Up to `steps_per_checkpoint` stale | Fresh — written during the grace window |
| Lost steps per slice | ~800 | ≤ 1 |
| Resubmission | Manual | Reconciler does it within 30 s |
| User-facing timeout | Silently clamped to `max_train_time` | Honored across slices |

The fix is small. The arrangement is the point: a per-slice scheduler limit (which the cluster wants for fairness and discipline), a clean checkpoint at the boundary (which the trainer wants for correctness), and a generic reconciler that doesn't know about any of it (which the operator wants for maintainability).

If you're running long jobs on ScalarLM, the things you can tune:

- `train_args["timeout"]` — your total wall-clock budget. Set it to however long you want the run to last.
- `steps_per_checkpoint` — frequency of mid-slice checkpoints. Smaller = less work lost on a hard crash, more I/O during training. The SIGTERM handler reduces the cost of large values for the slurm-timeout case specifically, but mid-slice crashes still fall back to this.
- `signal_grace_seconds` (server config) — increase if your checkpoint writes routinely exceed 5 minutes.
- `max_train_time` (server config) — your cluster's per-job cap. Keep it small enough that operational issues surface, large enough that overhead from the rotation stays low.
