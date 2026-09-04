from cray_megatron.megatron.dataset.data_loader import DataLoader

from cray_megatron.models.get_model_manager import get_model_manager
from cray_megatron.models.does_any_checkpoint_exist import does_any_checkpoint_exist
from cray_megatron.models.get_latest_checkpoint_path import (
    get_latest_checkpoint_path,
    delete_old_checkpoints,
)

from cray_megatron.collectives.main_rank_only import main_rank_only, is_main_rank
from cray_megatron.megatron.training_harness import TrainingHarness
from cray_megatron.megatron import stop_flag
from cray_megatron.megatron.diffusion_corruption import corrupt_canvas
from cray_megatron.megatron.dataset.diffusion_canvas import anchor_token_id
from cray_megatron.megatron.determinism import apply_seed
from cray_megatron.megatron.doc_mask import (
    doc_mask_decision,
    is_multimodal,
    is_diffusion,
    BUILD,
    SKIP_SEQLEN,
    SKIP_MULTIMODAL,
    SKIP_SSM,
)

from cray_infra.training.training_job_status import TrainingJobStatus
from cray_infra.util.get_job_config import get_job_config


from torch.optim import AdamW, SGD, RMSprop
import torch.nn.functional as F

import torch

import time
import logging
from cray_infra.training.distributed import allreduce, get_size, get_rank
from cray_infra.training.train_debug import is_train_debug_enabled


def _trace_loop(msg: str) -> None:
    """CRAY_TRAIN_DEBUG tracing for the training loop's phase boundaries."""
    if not is_train_debug_enabled():
        return
    import os as _os
    import sys as _sys
    import time as _time

    rank = _os.environ.get("RANK", _os.environ.get("SLURM_PROCID", "?"))
    _sys.stderr.write(f"[rank={rank}] training_loop [{_time.monotonic():.3f}]: {msg}\n")
    _sys.stderr.flush()

try:
    from torch.nn.attention.flex_attention import BlockMask
    _FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    _FLEX_ATTENTION_AVAILABLE = False

logger = logging.getLogger(__name__)

# Sequence-length cap for the SDPA path's materialized [B, 1, S, S] document mask.
_MAX_4D_MASK_SEQ_LEN = 16384
_doc_mask_skip_warned = False
_doc_mask_multimodal_warned = False
_doc_mask_ssm_warned = False


def _warn_doc_mask_skipped(seq_len: int) -> None:
    global _doc_mask_skip_warned
    if _doc_mask_skip_warned:
        return
    _doc_mask_skip_warned = True
    logger.warning(
        "Skipping 4D document mask: seq_len=%d exceeds cap %d. "
        "Packed documents will attend across each other in this run. "
        "Set attn_implementation='flex_attention' in train_args to lift this limit.",
        seq_len, _MAX_4D_MASK_SEQ_LEN,
    )


def _warn_doc_mask_skipped_multimodal() -> None:
    global _doc_mask_multimodal_warned
    if _doc_mask_multimodal_warned:
        return
    _doc_mask_multimodal_warned = True
    logger.warning(
        "Skipping 4D document mask: multimodal wrapper masks its loss by a 2D "
        "attention_mask (e.g. Gemma3ForConditionalGeneration indexes shift_logits "
        "by it), so a 4D mask raises IndexError. Falling back to the 2D mask; "
        "packed documents will attend across each other in this run."
    )


def _warn_doc_mask_skipped_ssm() -> None:
    global _doc_mask_ssm_warned
    if _doc_mask_ssm_warned:
        return
    _doc_mask_ssm_warned = True
    logger.warning(
        "Skipping 4D document mask: hybrid Mamba/SSM model. Its kernel-less "
        "torch_forward mixer multiplies hidden states by a 2D padding mask, so a "
        "4D mask broadcasts to 5D and crashes (NemotronH). Falling back to the 2D "
        "mask; packed documents will attend across each other in this run. (The "
        "flex_attention BlockMask can't be consumed by the SSM path either.)"
    )


def _use_flex_attention() -> bool:
    return get_job_config().get("attn_implementation") == "flex_attention"


def _build_document_block_mask(doc_ids, device):
    """Build a BlockMask for packed-document causal attention."""
    try:
        from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE as BLOCK
    except ImportError:
        BLOCK = 128

    B, S = doc_ids.shape
    num_blocks = (S + BLOCK - 1) // BLOCK
    padded_S   = num_blocks * BLOCK

    # Pad to block boundary with a sentinel doc_id that never matches any real doc.
    if padded_S > S:
        sentinel   = int(doc_ids.max().item()) + 1
        pad        = torch.full((B, padded_S - S), sentinel, dtype=doc_ids.dtype, device=device)
        doc_ids_p  = torch.cat([doc_ids, pad], dim=1)
    else:
        doc_ids_p = doc_ids

    # Per-block doc-ID range: min and max of the BLOCK tokens in each block.
    d     = doc_ids_p.view(B, num_blocks, BLOCK)
    d_min = d.min(dim=-1).values   # [B, num_blocks]
    d_max = d.max(dim=-1).values   # [B, num_blocks]

    blk = torch.arange(num_blocks, device=device)

    # Causal at block level: kv_b <= q_b admits at least one valid token pair.
    causal_any      = blk.view(1, -1) <= blk.view(-1, 1)   # [num_q, num_kv]
    strictly_causal = blk.view(1, -1) <  blk.view(-1, 1)   # [num_q, num_kv]

    # Doc overlap: ranges [d_min_kv, d_max_kv] and [d_min_q, d_max_q] intersect.
    d_min_q  = d_min.unsqueeze(2)   # [B, num_q, 1]
    d_max_q  = d_max.unsqueeze(2)   # [B, num_q, 1]
    d_min_kv = d_min.unsqueeze(1)   # [B, 1, num_kv]
    d_max_kv = d_max.unsqueeze(1)   # [B, 1, num_kv]
    doc_overlap = (d_min_kv <= d_max_q) & (d_min_q <= d_max_kv)   # [B, num_q, num_kv]

    # non_empty: block (q_b, kv_b) has at least one valid attention connection.
    non_empty = causal_any.unsqueeze(0) & doc_overlap   # [B, num_q, num_kv]

    # is_full: no per-token masking needed — kv strictly before q, one doc.
    q_single  = (d_min_q  == d_max_q)    # [B, num_q, 1]
    kv_single = (d_min_kv == d_max_kv)   # [B, 1, num_kv]
    same_doc  = (d_min_q  == d_min_kv)   # [B, num_q, num_kv]
    is_full   = strictly_causal.unsqueeze(0) & q_single & kv_single & same_doc

    def _pack(mask_bqkv):
        """Pack a [B, nq, nkv] bool mask into (counts [B,1,nq], indices [B,1,nq,max_k])."""
        Bm, nq, nkv = mask_bqkv.shape
        counts = mask_bqkv.sum(dim=2).to(torch.int32)   # [B, nq]
        max_k  = int(counts.max().item()) if counts.numel() > 0 else 0
        if max_k == 0:
            return counts.unsqueeze(1), torch.zeros(Bm, 1, nq, 0, dtype=torch.int32, device=device)
        all_idx = torch.arange(nkv, device=device).view(1, 1, nkv).expand(Bm, nq, nkv)
        # Replace invalid entries with sentinel nkv so they sort after valid ones.
        filled  = torch.where(mask_bqkv, all_idx, torch.full_like(all_idx, nkv))
        sorted_idx, _ = filled.sort(dim=2)
        # Take the first max_k entries; clamp out-of-range padding to a safe index.
        idx = sorted_idx[:, :, :max_k].clamp(0, nkv - 1).to(torch.int32)
        return counts.unsqueeze(1), idx.unsqueeze(1)

    kv_num_blocks,      kv_indices      = _pack(non_empty)
    full_kv_num_blocks, full_kv_indices = _pack(is_full)
    q_num_blocks,       q_indices       = _pack(non_empty.transpose(1, 2))
    full_q_num_blocks,  full_q_indices  = _pack(is_full.transpose(1, 2))

    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (doc_ids[b, q_idx] == doc_ids[b, kv_idx])

    try:
        return BlockMask(
            kv_num_blocks=kv_num_blocks,
            kv_indices=kv_indices,
            full_kv_num_blocks=full_kv_num_blocks,
            full_kv_indices=full_kv_indices,
            q_num_blocks=q_num_blocks,
            q_indices=q_indices,
            full_q_num_blocks=full_q_num_blocks,
            full_q_indices=full_q_indices,
            BLOCK_SIZE=(BLOCK, BLOCK),
            mask_mod=mask_mod,
            seq_lengths=(S, S),
        )
    except TypeError:
        # seq_lengths not present in older PyTorch builds.
        return BlockMask(
            kv_num_blocks=kv_num_blocks,
            kv_indices=kv_indices,
            full_kv_num_blocks=full_kv_num_blocks,
            full_kv_indices=full_kv_indices,
            q_num_blocks=q_num_blocks,
            q_indices=q_indices,
            full_q_num_blocks=full_q_num_blocks,
            full_q_indices=full_q_indices,
            BLOCK_SIZE=(BLOCK, BLOCK),
            mask_mod=mask_mod,
        )


class TrainingLoop:
    def __init__(self, training_harness: TrainingHarness):
        self.training_harness = training_harness

        self.callbacks = get_callbacks(self)

        self.training_state = TrainingState()

    def train(self):
        # Must run before load_model() so every global-RNG draw is deterministic.
        apply_seed(get_job_config().get("seed"))

        self.model_manager = get_model_manager()

        self.training_state.model_info = self.model_manager.load_model()

        self.training_loop()

        self.checkpoint()

        self._finalize_slice()

    def _finalize_slice(self):
        """Persist accumulated wall time after every slice."""
        slice_elapsed = time.time() - self.training_state.start_time
        accumulated = (
            self.training_state.accumulated_seconds_at_slice_start + slice_elapsed
        )
        self._persist_accumulated_seconds(accumulated)

    @main_rank_only
    def _persist_accumulated_seconds(self, accumulated_seconds):
        # Read-modify-write through the harness; never change status here.
        current = self.training_harness.get_status()
        self.training_harness.update_status(
            status=current.get("status", TrainingJobStatus.TRAINING),
            metadata={"accumulated_train_seconds": accumulated_seconds},
        )

    def training_loop(self):
        self.on_train_begin()

        self._load_accumulated_seconds()

        self.training_state.model_info["model"].train()

        max_steps = get_max_steps()
        gradient_accumulation_steps = get_gradient_accumulation_steps()

        self.training_state.optimizer = get_optimizer(
            self.training_state.model_info["model"]
        )
        self.training_state.scheduler = get_scheduler(
            self.training_state.optimizer, max_steps
        )

        if does_any_checkpoint_exist():
            self.resume_from_checkpoint()

        data_loader = DataLoader(
            model=self.training_state.model_info["model"],
            tokenizer=self.training_state.model_info["tokenizer"],
            starting_epoch=self.training_state.epoch,
        )

        data_iterator = iter(data_loader)

        # IterableDataset has no seek, so pull and discard `data_cursor` batches.
        if self.training_state.data_cursor > 0:
            logger.info(
                "Resuming dataloader: skipping %d batches into epoch %d",
                self.training_state.data_cursor,
                self.training_state.epoch,
            )
            for _ in range(self.training_state.data_cursor):
                next(data_iterator)

        starting_step = self.training_state.current_step

        self.print_device_info()

        for step in range(starting_step, max_steps):
            self.training_state.current_step = step
            self.training_state.epoch = data_loader.epoch

            step_start_time = time.time()

            self.on_step_begin(step)

            # accumulated_loss stays a device tensor; reduced once after the loop.
            accumulated_loss = None
            self.training_state.optimizer.zero_grad()

            for accum_step in range(gradient_accumulation_steps):
                prev_epoch = data_loader.epoch
                batch = next(data_iterator)

                # Update epoch if it changed during accumulation
                self.training_state.epoch = data_loader.epoch

                # Track cursor for checkpoint resume; reset to 1 on epoch rollover.
                if data_loader.epoch != prev_epoch:
                    self.training_state.data_cursor = 1
                else:
                    self.training_state.data_cursor += 1

                loss = self.training_step_accumulate(
                    batch,
                    accum_step,
                    gradient_accumulation_steps
                )

                # Device-side accumulation: no host sync, no collective.
                accumulated_loss = (
                    loss if accumulated_loss is None else accumulated_loss + loss
                )

            # Ensure gradients are synchronized across ranks during backward pass
            self.training_state.model_info["model"].backward_sync()

            # Clips, checks finiteness on every rank, and steps only if finite.
            stepped = self.optimizer_step()

            # Reporting-only reduction; must stay after the step or FSDP2 deadlocks.
            avg_accumulated_loss = float("nan")
            if accumulated_loss is not None:
                _, avg_accumulated_loss = self.sync_loss(
                    accumulated_loss / gradient_accumulation_steps
                )

            if not stepped:
                logger.warning(
                    f"Non-finite gradients at step {step} (avg loss="
                    f"{avg_accumulated_loss}) — optimizer step skipped"
                )
                self.training_state.nan_steps += 1
                avg_accumulated_loss = float('nan')
            else:
                # Log the averaged loss
                self.update_history(avg_accumulated_loss)

            # Print training step info with averaged loss
            step_time = time.time() - step_start_time
            self.print_training_step_info(avg_accumulated_loss, step_time)

            self.on_step_end(step)

            if stop_flag.was_stop_requested():
                logger.info(
                    "Stop requested via signal %s — exiting training loop "
                    "at step %d to checkpoint cleanly",
                    stop_flag.last_signal(),
                    step,
                )
                self.training_state.should_stop_training = True

            if self._stop_requested_on_any_rank(
                self.training_state.should_stop_training
            ):
                self.training_state.should_stop_training = True
                break

        self.on_train_end()

    def _stop_requested_on_any_rank(self, local_stop):
        """True when ANY rank wants to stop. Collective — every rank must call."""
        group = self._loss_process_group()
        if group is None:
            return bool(local_stop)

        import torch.distributed as dist

        device = self.training_state.model_info["distribution_strategy"]["device"]
        flag = torch.tensor([1.0 if local_stop else 0.0], device=device)
        dist.all_reduce(flag, group=group)
        return flag.item() > 0.0

    def _load_accumulated_seconds(self):
        # Sum of wall-time across prior slices, persisted by _finalize_slice.
        status = self.training_harness.get_status()
        self.training_state.accumulated_seconds_at_slice_start = float(
            status.get("accumulated_train_seconds", 0.0)
        )
        if self.training_state.accumulated_seconds_at_slice_start > 0:
            logger.info(
                "Resuming with %.0fs of prior training already elapsed",
                self.training_state.accumulated_seconds_at_slice_start,
            )

    def resume_from_checkpoint(self):
        latest_checkpoint_path = get_latest_checkpoint_path()
        logger.info(f"Resuming from checkpoint {latest_checkpoint_path}")

        checkpoint = torch.load(latest_checkpoint_path, weights_only=True)

        # `step` is the step that COMPLETED when the save fired, so resume at the next.
        self.training_state.current_step = checkpoint["step"] + 1
        self.training_state.epoch = checkpoint["epoch"]
        self.training_state.nan_steps = checkpoint.get("nan_steps", 0)
        # .get() with default=0 keeps older checkpoints (pre-Fix 2) loadable.
        self.training_state.data_cursor = checkpoint.get("data_cursor", 0)
        model = self.training_state.model_info["model"]
        if hasattr(model, "load_unwrapped_model"):
            # FSDP: re-shard the gathered checkpoint tensors into this rank.
            model.load_unwrapped_model(checkpoint["model_state_dict"])
        else:
            _load_trained_parameters(model, checkpoint["model_state_dict"])
        # Same for the optimizer state: re-shard rather than load wholesale.
        if hasattr(model, "load_unwrapped_optimizer"):
            model.load_unwrapped_optimizer(
                self.training_state.optimizer, checkpoint["optimizer_state_dict"]
            )
        else:
            self.training_state.optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )
        self.training_state.scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        # Restore RNG so the next forward is bit-identical; set_rng_state needs CPU.
        rng_state = checkpoint.get("rng_state")
        if rng_state is not None:
            torch.set_rng_state(rng_state.cpu())
        # cuda_rng_state holds one entry PER RANK; restore only this rank's device.
        import torch.distributed as dist

        cuda_rng_state = checkpoint.get("cuda_rng_state")
        if cuda_rng_state and torch.cuda.is_available():
            rank = get_rank() if (dist.is_available() and dist.is_initialized()) else 0
            if rank < len(cuda_rng_state):
                torch.cuda.set_rng_state(
                    cuda_rng_state[rank].cpu(), torch.cuda.current_device()
                )
            else:
                # Checkpoint from an older/narrower run — leave the seeded RNG alone.
                logger.warning(
                    "cuda_rng_state has %d entries but this is rank %d; "
                    "leaving this rank's CUDA RNG at its seeded value.",
                    len(cuda_rng_state),
                    rank,
                )

        self.training_state.history = self.training_harness.get_status()["history"]

    def training_step_accumulate(self, batch, accum_step, gradient_accumulation_steps):
        """Perform a single forward/backward pass with gradient accumulation."""
        model_config = self.training_state.model_info.get("model_config")

        # DiffusionGemma has a wholly different forward contract; own path below.
        if is_diffusion(model_config):
            return self._diffusion_training_step_accumulate(
                batch, accum_step, gradient_accumulation_steps
            )

        # Sequence classification: labels are class indices, no packed-doc masking.
        if get_job_config()["training_mode"] == "classification":
            return self._classification_training_step_accumulate(
                batch, accum_step, gradient_accumulation_steps
            )

        # Embedding training: sentence pairs with a CoSENT objective, no logits.
        if get_job_config()["training_mode"] == "embedding":
            return self._embedding_training_step_accumulate(
                batch, accum_step, gradient_accumulation_steps
            )

        device = self.training_state.model_info["distribution_strategy"]["device"]

        start_time = time.time()

        forward_kwargs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "labels": batch["labels"].to(device),
        }

        # Packed batches swap the 2D padding mask for a 4D block-diagonal+causal one.
        seq_len = forward_kwargs["input_ids"].shape[-1]
        model_config = self.training_state.model_info.get("model_config")
        decision = doc_mask_decision(batch, seq_len, model_config, _MAX_4D_MASK_SEQ_LEN)
        use_flex = _use_flex_attention() and _FLEX_ATTENTION_AVAILABLE
        if decision == SKIP_MULTIMODAL:
            # Wrapper indexes logits by a 2D mask; a 4D mask would IndexError.
            _warn_doc_mask_skipped_multimodal()
        elif decision == SKIP_SSM:
            # Hybrid Mamba/SSM mixer assumes a 2D mask; a 4D one broadcasts to 5D.
            _warn_doc_mask_skipped_ssm()
        elif decision == BUILD or (decision == SKIP_SEQLEN and use_flex):
            # The flex BlockMask is O(S/128), so _MAX_4D_MASK_SEQ_LEN doesn't apply.
            doc_ids = batch["document_ids"].to(device)
            if use_flex:
                # flex_attention path: O(S/128) BlockMask, no sequence-length cap.
                forward_kwargs["attention_mask"] = _build_document_block_mask(doc_ids, device)
            else:
                # SDPA path: materialize the [B, 1, S, S] causal block-diagonal mask.
                same_doc = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)
                causal = torch.ones(
                    seq_len, seq_len, device=device, dtype=torch.bool
                ).tril()
                forward_kwargs["attention_mask"] = (same_doc & causal).unsqueeze(1)
            forward_kwargs["position_ids"] = batch["position_ids"].to(device)
        elif decision == SKIP_SEQLEN:
            _warn_doc_mask_skipped(seq_len)
        # decision == NONE: batch isn't packed; leave the 2D mask as-is.

        # Multimodal wrappers require mm_token_type_ids even on text-only batches.
        if is_multimodal(model_config):
            forward_kwargs["mm_token_type_ids"] = torch.zeros_like(
                forward_kwargs["input_ids"]
            )

        # forward pass
        loss = self.training_state.model_info["model"](**forward_kwargs).loss

        # Scale loss to account for accumulation
        scaled_loss = loss / gradient_accumulation_steps

        # No cross-rank reduction here: the caller issues it after the accum loop.

        # Always call backward, even on NaN/Inf: it frees the saved activations.
        scaled_loss.backward()

        # Log info for each micro-batch
        self.print_microbatch_info(accum_step, loss, start_time)

        # Local, unreduced, detached. The caller sums these and reduces once.
        return loss.detach()

    def _embedding_training_step_accumulate(
        self, batch, accum_step, gradient_accumulation_steps
    ):
        """Embedding training step: CoSENT loss over a sentence pair."""
        device = self.training_state.model_info["distribution_strategy"]["device"]
        start_time = time.time()

        sentence1_features = {
            key.replace("sentence1_", ""): batch[key].to(device)
            for key in batch
            if key.startswith("sentence1_")
        }
        sentence2_features = {
            key.replace("sentence2_", ""): batch[key].to(device)
            for key in batch
            if key.startswith("sentence2_")
        }

        loss = self.training_state.model_info["loss"](
            sentence_features=[sentence1_features, sentence2_features],
            labels=batch["labels"].to(device),
        )

        scaled_loss = loss / gradient_accumulation_steps
        scaled_loss.backward()

        self.print_microbatch_info(accum_step, loss, start_time)
        # Local, unreduced, detached. The caller sums these and reduces once.
        return loss.detach()

    def _classification_training_step_accumulate(
        self, batch, accum_step, gradient_accumulation_steps
    ):
        """Sequence-classification training step."""
        device = self.training_state.model_info["distribution_strategy"]["device"]
        classification = get_job_config().get("classification") or {}
        label_smoothing = classification.get("label_smoothing", 0.0) or 0.0

        start_time = time.time()

        outputs = self.training_state.model_info["model"](
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )

        if label_smoothing > 0:
            logits = outputs.logits
            targets = batch["labels"].to(device)
            log_probs = F.log_softmax(logits, dim=-1)
            num_labels = logits.size(-1)

            smooth_targets = torch.zeros_like(log_probs).scatter_(
                -1, targets.unsqueeze(-1), 1.0
            )
            smooth_targets = (
                smooth_targets * (1.0 - label_smoothing) + label_smoothing / num_labels
            )
            loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        else:
            loss = outputs.loss

        scaled_loss = loss / gradient_accumulation_steps

        # Unconditional backward: it frees the saved activations even on NaN.
        scaled_loss.backward()

        self.print_microbatch_info(accum_step, loss, start_time)

        # Local, unreduced, detached. The caller sums these and reduces once.
        return loss.detach()

    def _diffusion_training_step_accumulate(
        self, batch, accum_step, gradient_accumulation_steps
    ):
        """DiffusionGemma canvas-denoising training step."""
        device = self.training_state.model_info["distribution_strategy"]["device"]
        model = self.training_state.model_info["model"]
        model_config = self.training_state.model_info.get("model_config")

        start_time = time.time()

        encoder_input_ids = batch["encoder_input_ids"].to(device)
        encoder_attention_mask = batch["encoder_attention_mask"].to(device)
        canvas_input_ids = batch["canvas_input_ids"].to(device)  # clean, pad-filled
        canvas_labels = batch["canvas_labels"].to(device)        # clean, -100 on pad

        vocab_size = model_config.text_config.vocab_size
        eps = self._diffusion_eps()

        # Corruption is resampled every step; protect_prefix keeps the anchor clean.
        protect_prefix = 1 if self._diffusion_anchor_id() is not None else 0

        # NaRA: the noise level must reach the hypernetwork before any forward pass.
        nara_context = self._diffusion_nara_context()
        if nara_context is not None:
            decoder_input_ids, noise_level = corrupt_canvas(
                canvas_input_ids, canvas_labels, vocab_size, eps,
                protect_prefix=protect_prefix, return_noise_level=True,
            )
            nara_context.set_noise_level(noise_level)
        else:
            decoder_input_ids = corrupt_canvas(
                canvas_input_ids, canvas_labels, vocab_size, eps,
                protect_prefix=protect_prefix,
            )

        base_kwargs = {
            "input_ids": encoder_input_ids,
            "attention_mask": encoder_attention_mask,
            "decoder_input_ids": decoder_input_ids,
        }
        # Pass mm_token_type_ids defensively: the encoder can crash without it.
        if is_multimodal(model_config):
            base_kwargs["mm_token_type_ids"] = torch.zeros_like(encoder_input_ids)

        # Self-conditioning: a no-grad prediction fed back on the gradient pass.
        sc_prob = self._diffusion_self_conditioning_prob()
        if sc_prob > 0.0:
            with torch.no_grad():
                sc_logits = model(
                    **base_kwargs,
                    self_conditioning_logits=None,
                    self_conditioning_mask=None,
                ).logits.detach()
            sc_mask = torch.rand(decoder_input_ids.size(0), device=device) < sc_prob
            forward_kwargs = {
                **base_kwargs,
                "self_conditioning_logits": sc_logits,
                "self_conditioning_mask": sc_mask,
            }
        else:
            forward_kwargs = {
                **base_kwargs,
                "self_conditioning_logits": None,
                "self_conditioning_mask": None,
            }

        outputs = model(**forward_kwargs)
        logits = outputs.logits  # (B, canvas_length, vocab)

        # fp32 CE for numerical stability; ignore padded canvas slots.
        canvas_loss_weight = batch.get("canvas_loss_weight")
        if canvas_loss_weight is not None:
            per_position = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                canvas_labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            )
            weights = canvas_loss_weight.to(device).reshape(-1).to(per_position.dtype)
            loss = (per_position * weights).sum() / weights.sum().clamp_min(1e-8)
        else:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                canvas_labels.reshape(-1),
                ignore_index=-100,
            )

        scaled_loss = loss / gradient_accumulation_steps
        scaled_loss.backward()

        self.print_microbatch_info(accum_step, loss, start_time)
        # Local, unreduced, detached. The caller sums these and reduces once.
        return loss.detach()

    def _diffusion_nara_context(self):
        """Return the model's NaRAContext when NaRA is enabled, else None."""
        if getattr(self, "_nara_context_resolved", False):
            return self._nara_context_cache

        self._nara_context_resolved = True
        self._nara_context_cache = None

        job_config = get_job_config()
        diffusion = job_config.get("diffusion") or {}
        nara = diffusion.nara if hasattr(diffusion, "nara") else diffusion.get("nara")
        if nara is not None:
            enabled = nara.enabled if hasattr(nara, "enabled") else nara.get("enabled", False)
            if enabled:
                from adapters.nara_prototype import find_nara_context
                self._nara_context_cache = find_nara_context(
                    self.training_state.model_info["model"]
                )
        return self._nara_context_cache

    def _diffusion_eps(self):
        job_config = get_job_config()
        diffusion = job_config.get("diffusion") or {}
        if hasattr(diffusion, "eps"):
            return diffusion.eps
        return diffusion.get("eps", 0.001)

    def _diffusion_anchor_id(self):
        """Resolve the Tier-2 canvas anchor id when enabled, else None."""
        job_config = get_job_config()
        diffusion = job_config.get("diffusion") or {}
        if hasattr(diffusion, "anchor_token"):
            enabled = bool(diffusion.anchor_token)
        else:
            enabled = bool(diffusion.get("anchor_token", False))
        if not enabled:
            return None
        tokenizer = self.training_state.model_info["tokenizer"]
        return anchor_token_id(tokenizer)

    def _diffusion_self_conditioning_prob(self):
        """Per-step probability of self-conditioning during training (0 disables)."""
        job_config = get_job_config()
        diffusion = job_config.get("diffusion") or {}
        if hasattr(diffusion, "self_conditioning_prob"):
            return diffusion.self_conditioning_prob
        return diffusion.get("self_conditioning_prob", 0.5)

    def optimizer_step(self):
        """Clip gradients, verify they are finite, then step; True if it stepped."""
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.training_state.model_info["model"].parameters(),
            get_gradient_clip_value(),
        )

        if not self._grads_finite_on_all_ranks(total_norm):
            return False

        self.training_state.optimizer.step()
        self.training_state.scheduler.step()
        return True

    def _grads_finite_on_all_ranks(self, total_norm):
        """True only when EVERY rank's gradients are finite. Collective."""
        local_bad = 0.0 if bool(torch.isfinite(total_norm)) else 1.0

        group = self._loss_process_group()
        if group is None:
            return local_bad == 0.0

        import torch.distributed as dist

        device = self.training_state.model_info["distribution_strategy"]["device"]
        flag = torch.tensor([local_bad], device=device)
        dist.all_reduce(flag, group=group)
        return flag.item() == 0.0

    def _loss_process_group(self):
        """A process group for this module's reductions only. Collective — all ranks."""
        if get_size() <= 1:
            return None
        if getattr(self, "_loss_pg", None) is None:
            import torch.distributed as dist

            self._loss_pg = dist.new_group()
            logger.info("Created dedicated process group for loss reduction")
        return self._loss_pg

    def sync_loss(self, loss):
        """Mean of `loss` across ranks; never call between a forward and backward."""
        group = self._loss_process_group()
        if group is not None:
            import torch.distributed as dist

            # Reduce a COPY: all_reduce is in-place and `loss` is the caller's.
            reduced = loss.detach().clone()
            dist.all_reduce(reduced, group=group)
            avg_loss = reduced.item() / get_size()
        else:
            avg_loss = loss.item()

        return loss, avg_loss

    def on_train_begin(self):
        self.training_state.start_time = time.time()
        for callback in self.callbacks:
            if hasattr(callback, "on_train_begin"):
                callback.on_train_begin()

    def on_step_begin(self, step):
        for callback in self.callbacks:
            if hasattr(callback, "on_step_begin"):
                callback.on_step_begin(step)

    def on_step_end(self, step):
        for callback in self.callbacks:
            if hasattr(callback, "on_step_end"):
                callback.on_step_end(step)

    def on_train_end(self):

        logger.info(
            f"Training finished successfully after {time.time() - self.training_state.start_time} seconds"
        )
        if self.training_state.nan_steps > 0:
            logger.warning(
                f"Encountered {self.training_state.nan_steps} NaN steps during training"
            )
        for callback in self.callbacks:
            if hasattr(callback, "on_train_end"):
                callback.on_train_end()

    def checkpoint(self):
        model = self.training_state.model_info["model"]
        model_state_dict = {}
        if hasattr(model, "unwrap_model"):
            logger.info("Unwrapping model")
            model_state_dict = self.training_state.model_info["model"].unwrap_model()
        else:
            model_state_dict = filter_checkpoint(model.model, model.model.state_dict())

        # Gather the sharded optimizer state here, on every rank, and hand it down.
        optimizer = self.training_state.optimizer
        if hasattr(model, "unwrap_optimizer"):
            optimizer_state_dict = model.unwrap_optimizer(optimizer)
        else:
            optimizer_state_dict = optimizer.state_dict()

        # Same again for the CUDA RNG: one state per RANK, not per visible device.
        cuda_rng_state = self._gather_cuda_rng_state()

        self.save_checkpoint(model_state_dict, optimizer_state_dict, cuda_rng_state)

    def _gather_cuda_rng_state(self):
        """One CUDA RNG state per RANK (not per device). Collective — all ranks."""
        import torch.distributed as dist

        if not torch.cuda.is_available():
            return []

        local_state = torch.cuda.get_rng_state(torch.cuda.current_device())
        if not (dist.is_available() and dist.is_initialized()):
            return [local_state]

        device = torch.device("cuda", torch.cuda.current_device())
        staged = local_state.to(device)
        gathered = [torch.zeros_like(staged) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, staged)
        return [g.cpu() for g in gathered]

    @main_rank_only
    def save_checkpoint(self, model_state_dict, optimizer_state_dict, cuda_rng_state):

        checkpoint = {
            "model_state_dict": model_state_dict,
            # Gathered by the caller on every rank; this method is @main_rank_only.
            "optimizer_state_dict": optimizer_state_dict,
            "scheduler_state_dict": self.training_state.scheduler.state_dict(),
            "step": self.training_state.current_step,
            "epoch": self.training_state.epoch,
            "nan_steps": self.training_state.nan_steps,
            "data_cursor": self.training_state.data_cursor,
            "rng_state": torch.get_rng_state(),
            # One entry PER RANK, not get_rng_state_all(), which is per-device.
            "cuda_rng_state": cuda_rng_state,
            "metadata": build_adapter_metadata(),
        }

        checkpoint_name = f"checkpoint_{self.training_state.current_step}.pt"

        self.training_harness.checkpoint(
            checkpoint_state=checkpoint,
            checkpoint_name=checkpoint_name,
        )

        delete_old_checkpoints()

    @main_rank_only
    def update_history(self, loss):
        job_config = get_job_config()

        max_history_length = job_config["training_history_length"]

        entry = {
            "step": self.training_state.current_step,
            "loss": loss,
            # Wall-clock across ALL slices: start_time resets on every slice.
            "epoch": self.training_state.epoch,
            "time": (
                self.training_state.accumulated_seconds_at_slice_start
                + (time.time() - self.training_state.start_time)
            ),
        }

        self.training_state.history.append(entry)

        if len(self.training_state.history) > max_history_length:
            self.training_state.history = remove_closest_entry(
                self.training_state.history, max_history_length
            )

        self.training_harness.update_status(
            status=TrainingJobStatus.TRAINING,
            metadata={"history": self.training_state.history},
        )

    def print_training_step_info(self, loss, step_time):
        # Deliberately NOT @main_rank_only: its barriers cost collectives per step.
        if not is_main_rank():
            return
        logger.info(
            f"Training step {self.training_state.current_step} "
            f"- epoch {self.training_state.epoch} "
            f"- avg loss {loss:.4f} "
            f"- learning rate {self.training_state.scheduler.get_last_lr()[0]:.6f} "
            f"- step time {step_time:.3f} seconds"
        )

    def print_microbatch_info(self, accum_step, loss, start_time):
        # Deliberately NOT @main_rank_only — see print_training_step_info.
        if not is_main_rank():
            return

        # only log if there is more than one microbatch
        if get_gradient_accumulation_steps() <= 1:
            return

        # float() is a host sync, but local to rank 0 — no collective.
        logger.debug(
            f"  Microbatch {accum_step + 1} "
            f"- step {self.training_state.current_step} "
            f"- loss {float(loss):.4f} "
            f"- time {time.time() - start_time:.3f}s"
        )

    def print_device_info(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        idx = self.training_state.model_info["distribution_strategy"]["device"]
        logger.info(f"Using device {device}:{idx}")


def get_callbacks(trainer):
    return [
        TimeoutCallback(trainer),
        CheckpointCallback(trainer),
        CudaMemoryCallback(trainer),
    ]


class TimeoutCallback:
    """Stops training when the user's TOTAL budget across all slices is exhausted."""

    def __init__(self, trainer):
        self.trainer = trainer
        job_config = get_job_config()
        self.timeout = job_config["timeout"]
        self.start_time = time.time()

    def on_train_begin(self):
        pass

    def on_step_end(self, step):
        slice_elapsed = time.time() - self.start_time
        total_elapsed = (
            self.trainer.training_state.accumulated_seconds_at_slice_start
            + slice_elapsed
        )
        if total_elapsed > self.timeout:
            logger.info(
                "Training timed out after %.0fs total (%.0fs prior slices + "
                "%.0fs current slice) — user budget was %.0fs",
                total_elapsed,
                self.trainer.training_state.accumulated_seconds_at_slice_start,
                slice_elapsed,
                self.timeout,
            )
            self.trainer.training_state.should_stop_training = True


class CheckpointCallback:
    def __init__(self, trainer):
        self.trainer = trainer
        job_config = get_job_config()
        self.steps_per_checkpoint = job_config["steps_per_checkpoint"]

    def on_step_end(self, step):
        if step % self.steps_per_checkpoint == 0 and step != 0:
            start_time = time.time()
            self.trainer.checkpoint()
            logger.info(
                f"Checkpoint on step {step} took {time.time() - start_time} seconds"
            )


class CudaMemoryCallback:
    """Periodic snapshot of CUDA allocator state, every `cuda_memory_log_interval` steps."""

    def __init__(self, trainer):
        self.trainer = trainer
        job_config = get_job_config()
        self.interval = job_config.get("cuda_memory_log_interval", 100)

    def on_step_end(self, step):
        # NOT @main_rank_only: that decorator barriers on every step.
        if not is_main_rank():
            return
        if self.interval <= 0 or step % self.interval != 0:
            return
        if not torch.cuda.is_available():
            return
        gib = 1024 ** 3
        allocated = torch.cuda.memory_allocated() / gib
        reserved = torch.cuda.memory_reserved() / gib
        max_allocated = torch.cuda.max_memory_allocated() / gib
        logger.info(
            f"CUDA memory @ step {step}: "
            f"allocated={allocated:.2f} GiB, "
            f"reserved={reserved:.2f} GiB, "
            f"max_allocated={max_allocated:.2f} GiB"
        )


class TrainingState:
    def __init__(self):
        self.should_stop_training = False
        self.current_step = 0
        self.epoch = 0
        self.model_info = None
        self.optimizer = None
        self.scheduler = None
        self.history = []
        self.start_time = None
        self.nan_steps = 0
        # Batches consumed in the current epoch; checkpointed for resume.
        self.data_cursor = 0
        # Loaded from status.json at the start of every slice (0 on the first).
        self.accumulated_seconds_at_slice_start = 0.0


def get_max_steps():
    job_config = get_job_config()
    return job_config["max_steps"]


def get_gradient_accumulation_steps():
    job_config = get_job_config()
    return job_config.get("gradient_accumulation_steps", 4)


def get_optimizer(model):
    job_config = get_job_config()
    learning_rate = job_config["learning_rate"]
    # Only optimize trainable parameters (with LoRA, just the adapter weights).
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer_type = job_config.get("optimizer_type", "adamw")
    logger.info(
        f"Initializing {optimizer_type} optimizer for {len(trainable)} trainable parameters"
    )

    if optimizer_type == "adamw":
        return AdamW(trainable, lr=learning_rate)
    elif optimizer_type == "sgd":
        return SGD(trainable, lr=learning_rate)
    elif optimizer_type == "rmsprop":
        return RMSprop(trainable, lr=learning_rate)

    logger.warning(f"Unknown optimizer_type '{optimizer_type}', falling back to AdamW")
    return AdamW(trainable, lr=learning_rate)


def get_gradient_clip_value():
    job_config = get_job_config()
    return job_config.get("gradient_clip_value", 1.0)


def get_warmup_steps():
    job_config = get_job_config()
    return int(job_config.get("warmup_steps", 0))


def get_scheduler(optimizer, max_steps):
    warmup_steps = get_warmup_steps()
    if warmup_steps <= 0:
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=max_steps,
        )

    # Ramp from lr/1000 to lr over warmup_steps, then decay linearly to 0.
    decay_steps = max(1, max_steps - warmup_steps)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    decay = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=decay_steps,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, decay],
        milestones=[warmup_steps],
    )


def remove_closest_entry(history, max_length):
    # Drop the most closely spaced entries until the history fits max_length.
    while len(history) > max_length:
        min_diff = float("inf")
        min_index = None
        for i in range(1, len(history) - 1):
            diff = history[i + 1]["step"] - history[i - 1]["step"]
            if diff < min_diff:
                min_diff = diff
                min_index = i
        history.pop(min_index)

    return history


def filter_checkpoint(model, state_dict):
    # Remove the layers without gradients
    saved_params = {}

    for name, param in model.named_parameters(recurse=True):
        if param.requires_grad:
            logger.info(f"Saving parameter {name}")
            saved_params[name] = state_dict[name]

    return saved_params


def _load_trained_parameters(wrapped_model, state_dict):
    """Load a filtered (trainable-only) checkpoint back into the live model."""
    # Candidate load targets: the wrapper and its `.model` descendants.
    candidates = []
    module = wrapped_model
    seen = set()
    for _ in range(4):
        if module is None or id(module) in seen:
            break
        seen.add(id(module))
        candidates.append(module)
        module = getattr(module, "model", None)

    checkpoint_keys = set(state_dict)

    # Pick the target whose parameter namespace overlaps the checkpoint the most.
    best_target = None
    best_matched = -1
    for candidate in candidates:
        matched = len(checkpoint_keys & set(candidate.state_dict().keys()))
        if matched > best_matched:
            best_matched = matched
            best_target = candidate

    if best_matched <= 0:
        raise RuntimeError(
            f"Checkpoint resume could not align any of the {len(checkpoint_keys)} "
            f"saved parameter(s) with the live model. The model_state_dict "
            f"namespace matches no module in "
            f"{type(wrapped_model).__name__}(.model)*. Sharded FSDP checkpoints "
            f"need the reshard-on-load path; for the single-process / DDP / LoRA "
            f"paths this is a wrapper-namespace regression."
        )

    incompatible = best_target.load_state_dict(state_dict, strict=False)

    # missing_keys is expected (frozen weights); unexpected_keys means data loss.
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint resume left {len(incompatible.unexpected_keys)} trained "
            f"parameter(s) unloaded (unexpected keys), e.g. "
            f"{incompatible.unexpected_keys[:3]}. Refusing to continue with "
            f"partially-restored weights."
        )

    logger.info(
        "Restored %d trained parameter tensor(s) into %s on resume",
        best_matched,
        type(best_target).__name__,
    )
    return best_target


def build_adapter_metadata():
    """Build the `metadata` dict saved in the `.pt` alongside `model_state_dict`."""
    job_config = get_job_config()
    metadata: dict = {}

    if job_config.get("adapter_type") == "lora":
        lora_config = job_config.get("lora_config") or {}
        if "lora_alpha" in lora_config:
            metadata["lora_alpha"] = int(lora_config["lora_alpha"])
        if "use_rslora" in lora_config:
            metadata["use_rslora"] = bool(lora_config["use_rslora"])

    return metadata

class _AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # Perform allreduce operation out of place
        input_tmp = input.clone()
        allreduce(input_tmp)
        # Return the all-reduced tensor
        return input_tmp

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_output_tmp = grad_output.clone()
        # Perform allreduce operation in place
        allreduce(grad_output_tmp)
        # Return the all-reduced gradient
        return grad_output_tmp

allreduce_op = _AllReduce.apply
