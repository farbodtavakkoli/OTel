# Method: the GRPO loop

How GRPO is run in this repository: the loop itself, and the data, reward and adapter contract every
stack implements. Nothing here is specific to a model pair or a vendor. The two install routes live
in each stack's own `../training/llm/<framework>/grpo/README.md`, and what differs between ROCm and CUDA is in
[`grpo_porting.md`](grpo_porting.md).

## One iteration

GRPO is PPO without the value network. Its objective uses, per sampled token, the ratio
`π_new(y|x) / π_old(y|x)` clipped to `[1-ε, 1+ε]`, with the advantage of each completion measured
**relative to the other completions sampled for the same prompt**. That makes the unit of work small
and on-policy:

1. Take a batch of prompts.
2. For each prompt sample **G** completions from the **current** policy at one fixed temperature.
   The reference answer is not shown to the policy.
3. Reward every completion. Compute the group-relative advantage `A_i = (r_i − mean(r)) / std(r)`.
4. Do one gradient update on that batch.
5. Push the new weights to the inference engine.
6. Go to 1.

One "loop" is one batch and one update. An epoch over the prompt set is hundreds of loops. Nothing is
generated in bulk ahead of time: the policy that samples step *n* is the policy produced by step
*n − 1*.

## Why the loop has to be that small

The clipped ratio is only informative while the sampling policy and the policy being updated are
close. Generate a large batch of completions and train on it for many steps, and the ratio leaves the
clip window after the first few updates, so the gradient for those tokens goes to zero. Training on
stale rollouts is silently wasted compute, not a speed-up.

The same reasoning rules out training a fresh adapter on all collected rollouts at the end: those
came from a sequence of older policies, so the clip discards most of them. The artefact of GRPO is
the policy, not the rollout set. For a clean final artefact from curated data, use
**rejection-sampling fine-tuning** instead: keep the best-scored completion per prompt and SFT a
fresh adapter on it.

## Data

Prompts in chat format with a reference assistant answer. The reference is used by the reward
function (reference-grounded judging), never by the policy. The same file therefore serves an SFT
warm-up and the GRPO run. The contract is in the Data section of [`../training/llm/grpo_common/README.md`](../training/llm/grpo_common/README.md).

## Reward

The judge grades each completion **against the reference answer** and returns a score. Because
advantages are normalised within the group, only the within-group spread matters, not the absolute
scale, so a group whose G completions all score the same contributes zero gradient and is wasted.
Deterministic rewards run alongside the judge (format compliance, a length penalty); being
noise-free, they stabilise the first loops. No judge text is ever trained on.

The judge is one vLLM replica on its own GPU, shared by every stack so the reward is identical across
them. It is started by [`../training/llm/grpo_common/judge_server.sh`](../training/llm/grpo_common/judge_server.sh) and scored through
[`../training/llm/grpo_common/judge_reward.py`](../training/llm/grpo_common/judge_reward.py). Because it is shared, a stack's wait on it
grows with the number of runs pointed at it, so give each run its own judge when the per-stack split
matters. `GRPO_JUDGE_RUBRIC` selects the rubric, and changing it or the prompt template invalidates
comparison with existing numbers.

## Adapter handling

One LoRA adapter is created at the start and updated every loop, and is **not** merged into the base
model between loops: merging compounds low-rank updates into the base weights, accumulates bf16
rounding each time, and discards the free KL reference (the base model with the adapter disabled).
The inference engine hot-loads the adapter each loop, and it is merged once at the end for the
release artefact.

LoRA earns its place here for three infrastructure reasons: the KL reference costs no memory, the
per-loop sync moves the adapter rather than the whole base model, and vLLM can swap it without
reloading the base.

## Warm-up

Run an SFT pass over the same data before GRPO. It reduces zero-variance groups and format failures
in the first loops, which is where rollout compute is otherwise lost.
