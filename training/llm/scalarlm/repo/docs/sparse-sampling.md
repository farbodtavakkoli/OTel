# Design: Sparse Sequence Sampling for Long-Context Training

## Problem
Training on 128K sequences with $\text{head\_dim}=512$ on 24GB GPUs is currently impossible because the attention kernels (both `flex_attention` and `sdpa`) either materialize an $O(S^2)$ matrix or fall back to a dense implementation that attempts to allocate $\sim 128\text{GB}$ of memory.

The goal is to maintain the ability to train on long-context data without hitting OOM, while preserving the positional signals that allow the model to learn long-range dependencies.

## Proposed Solution: Global Block Budget Sampling
Instead of packing full documents into a sequence, we apply a "lossy compression" to the dataset and enforce a strict upper bound on the number of tokens per block. This ensures a deterministic memory profile regardless of document distribution.

### The "Global Budget" Strategy
We introduce a single configuration variable: `max_effective_block_size` (e.g., 32,768). This value serves as both the per-document sampling cap and the total per-block token limit.

### Sampling and Packing Logic
For each document in the dataset:
1. **Sparsification**: 
   - If `doc_len <= max_effective_block_size`: Keep the document in its entirety.
   - If `doc_len > max_effective_block_size`: Randomly sample exactly `max_effective_block_size` tokens from the document, preserving original relative order.
2. **Positional Integrity**: 
   - The original `position_ids` must be preserved. A token at index $100,000$ keeps `position_id=100,000` to maintain the global coordinates for RoPE.
3. **Budgeted Packing**: 
   - Concatenate these (potentially compressed) documents.
   - Slice the resulting stream into blocks of exactly `max_effective_block_size`.

### Theoretical Impact
- **Memory Guarantee**: The attention matrix size is strictly capped at $O(\text{max\_effective\_block\_size}^2)$. This eliminates the risk of the 128GB "dense fallback" OOM.
- **Positional Encoding**: RoPE continues to provide the model with the correct distance between tokens, even when the sequence is sparse.
- **Training Signal**: The model learns to attend to a sparse representation of the long context. By keeping the budget consistent, the model sees a stable density of tokens per block.

## Verification Plan
1. **Correctness**: Verify that `position_ids` in the packed blocks are non-contiguous for sampled documents.
2. **Memory**: Confirm that CUDA memory usage is stable and that no allocation requests exceed the 24GB GPU capacity.
3. **Convergence**: Monitor loss to ensure the sparse signal is sufficient for the model to converge.
