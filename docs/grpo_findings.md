## The recipe

- **Check the reward has variance first.** Advantages are computed within a group, so a group whose
  completions all score alike contributes no gradient. The usual cause is an integer judge scale,
  which collapses to near-bimodal; derive a continuous score from the judge's own log-probs instead,
  which costs nothing extra because the log-probs already come back with the call. While most groups
  are flat, no trainer or hyperparameter will help.

- **Warm up with SFT, then run GRPO.** Cover every source, since a single-source warm-up lifts that
  one and degrades the rest. Build the SFT sequence as the rollout prompt plus reference answer, and
  do it once at the start.

- **Keep the rest plain.** Use a moderate group size, temperature near 1.0, no group filtering and
  no KL penalty. Dropping the reference model also halves the GPUs a stack needs. Treat dynamic
  sampling as a throughput lever for a reward that is already informative, and note that it changes
  what a step means, so filtered and unfiltered runs are not step-comparable.

- **Measure on held-out prompts, greedily, never on train reward.** Train reward tracks batch
  composition rather than policy quality. Establish the eval set's noise band before believing a
  delta.

- **Size for the judge and the vocabulary.** Rollout plus judging is a fixed floor per step, and at
  smaller policy sizes the judge is the bottleneck, so budget roughly one judge replica per two
  training pairs. Memory peaks in the log-probability forward pass, driven by vocabulary size rather
  than parameter count, so micro-batch it with gradient checkpointing.

## Which stack to use

**ScalarLM and verl are the two to reach for.** Both distribute work across GPUs: verl puts a rollout
engine on every training GPU and syncs the adapter alone, while ScalarLM submits each step as a
scheduler job and hot-loads one adapter, carried across jobs, into the running server. Their per-step
times were close, so choose on how you want to operate GRPO: ScalarLM for GRPO inside a job scheduler
and serving stack, verl for the direct route with the most control.

### The others

Nothing below is broken. Each is fixed to a narrower layout, leaving less room to shape a run around
your hardware.

- **Red Hat training-hub** calls verl underneath at the same commit and layout, so it inherits verl's
  behaviour but not its flexibility: micro-batching is fixed by its defaults, with no override
  passthrough. Use it if you already run training-hub; otherwise call verl directly.

- **Axolotl, RapidFire AI and TRL + DeepSpeed** share TRL's GRPO core. Their split layout puts the
  rollout server on one GPU and the trainer on another, leaving the policy unsharded on a single
  training GPU, which is what makes them slower as the policy grows. Only Axolotl syncs the adapter
  alone; the others broadcast the merged policy each step.

- **Unsloth**'s colocated path shares weights between the trainer and an in-process vLLM engine, so
  rollouts cannot spread across GPUs. That is a property of this path, not of Unsloth generally.
  Avoid it for large multimodal policies, where the same path is known to produce corrupted
  generations.

- **OpenRLHF** needs the patch that ships with the stack. Without it the adapter never reaches the
  rollout engine, so training runs against a stale policy while appearing healthy.

- **TorchTitan RL** is full-parameter with no LoRA path and cannot run every architecture.

Per-stack setup, flags and known limitations are in each `../training/llm/<framework>/grpo/README.md`. Porting the
harness between ROCm and CUDA: [`grpo_porting.md`](grpo_porting.md).
