# Frequently Asked Questions

Full docs at [scalarlm.com](https://www.scalarlm.com).

---

## Training and Fine-tuning

### Do I need special prompt templates for training or inference?

Yes. Training and inference prompts must follow the same format as the prompt template for the model published on Hugging Face. Each deployment is tied to a specific model — refer to the model card for the prompt template.

### How can I monitor fine-tuning progress and loss curves?

Use the `scalarlm plot` CLI command after setting your API URL:

```bash
pip install scalarlm
export SCALARLM_API_URL="http://<your-deployment-ip>"
scalarlm plot
```

---

## Job Management

### What happens if I launch multiple fine-tuning jobs at once?

Jobs are queued automatically by the framework's built-in scheduler — you don't need to implement your own queue.

---

## Training Parameters

### How do I change the token-block (chunk) size?

Pass `max_token_block_size` in `train_args`:

```python
llm.train(
    dataset,
    train_args={
        "max_steps": 200,
        "learning_rate": 3e-4,
        "max_token_block_size": 1024
    }
)
```

### Can I change the quantization / dtype?

Yes, via `train_args`:

```python
llm.train(
    dataset,
    train_args={
        "max_steps": 200,
        "learning_rate": 3e-3,
        "gpus": 1,
        "dtype": "float32"
    }
)
```

At inference time, vLLM uses the dtype defined in the deployment configuration. When this differs from the dtype of a trained checkpoint, vLLM automatically converts the type when loading the model.

### Which GPU parameters should I use for multi-GPU training?

Use `gpus` to specify the total number of GPUs to request (e.g., `"gpus": 2`). Do not use `max_gpus` — that flag is for debugging only.

### Are there additional parameters needed for multi-GPU inference?

No. Multi-GPU inference is configured at deployment time via `values.yaml`.

---

## CLI Reference

### How do I install and configure the CLI?

```bash
pip install scalarlm
export SCALARLM_API_URL="http://<your-deployment-ip>"
```

Available commands:

```
scalarlm [-h] {logs,plot,ls,squeue} ...

positional arguments:
  logs      View training logs
  plot      Plot training loss curves
  ls        List available models
  squeue    View the job queue

options:
  -h, --help  show this help message and exit
```

### How do I modify training hyperparameters?

You can modify files in the `ml/` directory directly. For example, the optimizer configuration lives at:

```
ml/cray_megatron/megatron/training_loop.py
```

Set your `PYTHONPATH` to the checked-out repo's SDK before running:

```bash
export PYTHONPATH=/path/to/scalarlm/sdk
```

The `ml/` directory is packaged and shipped to the cluster automatically with each job submission — no Docker rebuild or redeployment required.

### Can I change the loss function?

Yes. Swap in a custom loss in the training loop at:

```
ml/cray_megatron/megatron/training_loop.py
```

See the [source on GitHub](https://github.com/tensorwavecloud/ScalarLM/blob/main/ml/cray_megatron/megatron/training_loop.py).

---

## Caching & Performance

### Does ScalarLM cache inference results?

No. Inference is sufficiently fast that no result cache is provided.

### What are the current known limitations?

- Each deployment is tied to a single base model.
- Large-model training may not yet be fully optimized; benchmarks and speedups are in progress.

---

## Advanced Topics

### Can I implement RLHF?

Yes. Use the ScalarLM inference endpoint to score or rank data with your reward model, then feed the selected data back into the training endpoint to update the model. The closed-loop design is a primary use case ScalarLM was built around.

### Is early stopping available?

The framework does not expose early-stop parameters directly, but because it is built on PyTorch and Hugging Face, you can integrate the [Hugging Face early stopping callback](https://huggingface.co/docs/transformers/en/main_classes/callback) into your training loop in the `ml/` directory.

### Where can I see a full list of fine-tuning parameters?

Rather than a single config file, ScalarLM lets you modify and write new code in the `ml/` directory to add or change training parameters. This gives you maximum flexibility to experiment without rebuilding infrastructure.

### Can I use any Hugging Face model?

In principle yes, but each deployment must be explicitly configured. See the supported and validated models in the [README](README.md) and on the [ScalarLM docs](https://www.scalarlm.com).

### How do I set inference temperature?

Temperature is a parameter passed directly to vLLM. The default is recommended — higher temperature increases output variance and error rate. If you need to change it, refer to the [vLLM quickstart docs](https://docs.vllm.ai/en/stable/getting_started/quickstart.html) for examples.

### What is the relationship between training steps and epochs in ScalarLM?

ScalarLM shifts away from classic batch-based training. Its definitions of "step" and "epoch" are chunk- and GPU-centric rather than example- and batch-driven.

**1. No conventional batching.** ScalarLM flattens all training examples into a single continuous token stream. There is no `batch_size` parameter.

**2. Token chunks (blocks).** The token stream is split into fixed-length chunks controlled by `max_token_block_size`. Each chunk contains the same number of tokens, so every GPU processes an identical workload per step. See the [chunking source](https://github.com/tensorwavecloud/ScalarLM/blob/main/ml/cray_megatron/megatron/dataset/load_dataset.py).

**3. Steps vs. epochs.**

- **Training step:** one forward-and-backward pass over one chunk on one GPU.
- **Epoch:** a complete pass through all chunks across all GPUs.

In traditional frameworks:
```
steps_per_epoch = ceil(num_examples ÷ batch_size)
```

In ScalarLM:
```
steps_per_epoch = total_chunks ÷ num_GPUs
```

**4. Shard-based parallelism.** The full set of chunks is partitioned into shards, one per GPU. Each GPU iterates through its shard sequentially, performing one training step per chunk.

**Key takeaway:** Steps are chunk-level iterations; epochs are full passes over all chunks divided by the number of GPUs. This guarantees uniform token throughput and balanced GPU workloads without a traditional batch size.

---

## Saving Models

### Can I save a fine-tuned model to Hugging Face?

Yes. See the [Save Fine-tuned Model to Hugging Face](https://www.scalarlm.com/save-fine-tuned-model-to-hugging-face/) guide for step-by-step instructions. Post-training checkpoints can also be pushed to the Hub automatically at the end of a training run.

---

*For more, visit [scalarlm.com](https://www.scalarlm.com) or open a [GitHub issue](https://github.com/tensorwavecloud/scalarlm/issues).*
