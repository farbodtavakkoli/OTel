"""Format a dataset locally and submit a training job to a remote ScalarLM server — see readme_scalarlm.md."""
import os
import json
import argparse
import scalarlm
import datetime
from dotenv import load_dotenv

load_dotenv('dev.env')


def _str2bool(value):
    """Parse a boolean flag; argparse's type=bool treats any non-empty string as True."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ('true', 't', 'yes', 'y', '1'):
        return True
    if normalized in ('false', 'f', 'no', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default = 'data/classification_data.json',
                        help='Local training data (JSON list or JSONL)')
    parser.add_argument('--custom_data_path', default=None, type=str,
                        help='Data path that already exists on the remote server/VM (overrides data_path)')
    parser.add_argument('--training_mode', default = 'language_model',
                        choices=['language_model', 'embedding', 'classification'])
    parser.add_argument('--model_type', default='qwen3',
                        help='Chat-template family for special-token formatting (gemma3, qwen3, llama3, '
                             'rnj-1, olmo3, mistral, gpt-oss_reasoning, gpt-oss_it, lfm, phi4)')
    parser.add_argument('--gradient_calculation', default='output_tokens',
                        choices=['output_tokens', 'entire_sequence'],
                        help='output_tokens: gradient only on output tokens (input masked); '
                             'entire_sequence: gradient on the full sequence')
    parser.add_argument('--sample_fraction', default=1.0, type=float,
                        help='Fraction of the dataset to use (0.0-1.0), useful for testing')

    # Training hyperparameters. Every numeric argument needs an explicit `type=`:
    # without it a passed flag stays a str and the server fails comparing it.
    parser.add_argument('--max_steps', default = 11, type=int)
    parser.add_argument('--learning_rate', default = 0.0005, type=float)
    parser.add_argument('--batch_size', default = 2, type=int)
    parser.add_argument('--max_token_block_size', default = 1762, type=int)

    # LoRA configuration
    parser.add_argument('--r', default = 8, type=int)
    parser.add_argument('--lora_alpha', default = 16, type=int)
    parser.add_argument('--lora_dropout', default = 0.05, type=float)
    parser.add_argument('--target_modules', default = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                        nargs='+',
                        choices=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    parser.add_argument('--use_rslora', default=False, type=_str2bool,
                        help='Rank-stabilized LoRA (alpha/sqrt(r) scaling); the merge step reads the same flag')
    parser.add_argument('--sampling_seed', default=42, type=int,
                        help='Dataset packing / replay RNG seed (distinct from --seed, which seeds torch/numpy)')

    parser.add_argument('--steps_per_checkpoint', default = 100, type=int)
    parser.add_argument('--adapter_type', default='none', choices=['lora', 'tokenformer', 'none'],
                        help='lora: recommended for most use cases; tokenformer: may need recipe '
                             'changes; none: full parameter training')
    parser.add_argument('--optimizer_type', default='adamw', choices=['adamw', 'sgd', 'rmsprop'],
                        help='New optimizers must also be added to the server-side training loop '
                             '(ml/cray_megatron/megatron/training_loop.py)')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--gradient_checkpointing', default=True, type=_str2bool)
    parser.add_argument('--distribution_strategy', default='fsdp',
                        choices=['fsdp', 'ddp', 'pytorch_fsdp'],
                        help='fsdp: shards model+data across GPUs (large models); '
                             'ddp: full model per GPU, data sharded (faster for small models); '
                             'pytorch_fsdp: PyTorch FSDP2 (fully_shard) — buckets and overlaps '
                             'collectives, so it scales better than fsdp at larger world sizes')
    # sdpa, not flash_attention_2: FA2 is unsupported for training on MI355X and
    # the server downgrades it to sdpa anyway.
    parser.add_argument('--attn_implementation', default='sdpa',
                        choices=['flash_attention_2', 'sdpa', 'eager'])
    parser.add_argument('--freeze_layer_keywords',
                        default="vision_model, vision_tower, multi_modal_projector, visual",
                        help='Comma-separated keywords of layers to freeze (e.g. vision encoders)')

    # Classification-specific settings
    parser.add_argument('--num_labels', default=None, type=int)
    parser.add_argument('--classification_dropout', default=0.1, type=float)
    parser.add_argument('--label_smoothing', default=0.1, type=float)

    # HuggingFace Hub upload settings
    parser.add_argument('--upload_to_hf', default=False, type=_str2bool)
    parser.add_argument('--hf_repo_id', default='farbodtavakkoli/scalarlm-test')
    parser.add_argument('--hf_upload_token', default=os.getenv('HF_TOKEN', ''))

    # gpus is GPUs per node. On a bare-metal 8-GPU box (MI355X) pass --gpus 8
    # --nodes 1; the server also needs SCALARLM_MAX_GPUS_PER_NODE=8, whose
    # default of 1 otherwise silently caps the job to a single GPU.
    parser.add_argument('--gpus', default = 1, type=int)
    parser.add_argument('--nodes', default = 2, type=int)

    return parser.parse_args()

# Special tokens formatting for different model types
def format_conversation(example, model_type):
    prompt = str(example.get('prompt', '')) 
    completion = str(example.get('completion', '')) 
    reasoning = str(example.get('reasoning', '')) 

    if model_type.lower() == "qwen3":
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>"

    elif model_type.lower() == "llama3":
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{completion}<|eot_id|>"

    elif model_type.lower() == "gemma3":
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{completion}<end_of_turn>"

    elif model_type.lower() == "rnj-1":
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{completion}<|eot_id|>"

    elif model_type.lower() == "olmo3":
        return f"<|endoftext|><|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|endoftext|>"

    elif model_type.lower() == "mistral":
        return f"<s>[SYSTEM_PROMPT][/SYSTEM_PROMPT][INST]{prompt}[/INST]{completion}</s>"

    elif model_type.lower() == "lfm":
        return f"<|startoftext|><|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>"

    elif model_type.lower() == "phi4":
        return f"<|user|>{prompt}<|end|><|assistant|>{completion}<|end|>"

    elif model_type.lower() == "gpt-oss_reasoning":
        return f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2026-02-11\n\nReasoning: low\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions\n\nreasoning_language: English\n\nYou are an Open Source Telecom Question answering model that uses provided contexts to answer telecom questions.\n\n<|end|><|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>analysis<|message|>{reasoning}<|end|><|start|>assistant<|channel|>final<|message|>{completion}<|return|>"

    elif model_type.lower() == "gpt-oss_it":
        return f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2026-02-11\n\nReasoning: low\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>final<|message|>{completion}<|return|>"

    else:
        return f"User: {prompt}\nAssistant: {completion}"

# Format example for training
def format_for_scalarlm(example, config):
    
    if config.gradient_calculation == 'output_tokens':
        # Split: put prompt in 'input' (masked), completion + reasoning in 'output' (gradient computed)
        prompt = str(example.get('prompt', '')) or ""
        completion = str(example.get('completion', '')) or ""
        reasoning = str(example.get('reasoning', '')) or ""

        # Format input portion with model-specific tokens (empty completion/reasoning)
        input_text = format_conversation({'prompt': prompt, 'completion': '', 'reasoning': ''}, model_type=config.model_type)
        
        # Format output portion with model-specific tokens (empty prompt)
        output_text = format_conversation({'prompt': '', 'completion': completion, 'reasoning': reasoning}, model_type=config.model_type)
        return {'input': input_text, 'output': output_text}
    elif config.gradient_calculation == 'entire_sequence':
        full_text = format_conversation(example, model_type=config.model_type)
        # Full sequence gradient: everything in 'output'
        return {'input': '', 'output': full_text}

# Loading data for either language model or embedding training
def get_dataset(config):
    import random
    SEED = 42
    dataset = []

    try:
        with open(config.data_path, 'r+') as f:
            data = json.load(f)
    except:
        with open(config.data_path, 'r+') as f:
            data = [json.loads(row) for row in f]

    # Subsample dataset if sample_fraction < 1.0
    if config.sample_fraction < 1.0:
        sample_size = int(len(data) * config.sample_fraction)
        print(f"Subsampling dataset to {config.sample_fraction:.2%} ({sample_size} samples)")
        random.seed(SEED)
        random.shuffle(data)
        data = data[:sample_size]

    if config.training_mode == 'embedding':
        for example in data:
            if 'sentence1' in example and 'sentence2' in example and 'score' in example:
                dataset.append({
                    'sentence1': example['sentence1'],
                    'sentence2': example['sentence2'],
                    'score': example['score']
                })
            else:
                # If data doesn't have correct format, skip
                print(f"Warning: Skipping example without sentence1/sentence2/score format: {example.keys()}")

    elif config.training_mode == 'classification':
        # Build label mapping from unique labels
        unique_labels = sorted(list(set(example.get('label', example.get('answer', '')) for example in data)))
        config.label2id = {label: i for i, label in enumerate(unique_labels)}
        config.id2label = {v: k for k, v in config.label2id.items()}
        config.num_labels = len(config.label2id)
        
        for example in data:
            # Support both 'text'/'label' and 'question'/'answer' column names
            text = example.get('text', example.get('question', ''))
            label = example.get('label', example.get('answer', ''))
            dataset.append({
                'text': text,
                'label': config.label2id[label]
            })

    elif config.training_mode == 'language_model':
        # For language model training
        for example in data:
            dataset.append(format_for_scalarlm(example, config))
    
    return dataset

def build_train_args(config):
    """Assemble the full job configuration passed to `llm.train()`."""
    return {
        # --- scheduling / shape ---
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "gpus": config.gpus,
        "nodes": config.nodes,
        "batch_size": config.batch_size,
        "max_token_block_size": config.max_token_block_size,
        "steps_per_checkpoint": config.steps_per_checkpoint,

        # --- mode / adapter ---
        "training_mode": config.training_mode,
        "adapter_type": config.adapter_type,

        # Must be nested: the server reads job_config["lora_config"][...].
        "lora_config": {
            "r": config.r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "target_modules": config.target_modules,
            "use_rslora": config.use_rslora,
        },
        "sampling_seed": config.sampling_seed,

        "optimizer_type": config.optimizer_type,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_checkpointing": config.gradient_checkpointing,
        "distribution_strategy": config.distribution_strategy,
        "attn_implementation": config.attn_implementation,
        "freeze_layer_keywords": [
            k.strip() for k in config.freeze_layer_keywords.split(',') if k.strip()
        ],
        "upload_to_hf": config.upload_to_hf,
        "hf_repo_id": config.hf_repo_id,
        "hf_upload_token": config.hf_upload_token,
        "custom_data_path": config.custom_data_path,

        # Must be nested; --classification_dropout maps to classification.dropout.
        "classification": {
            "num_labels": config.num_labels,
            "dropout": config.classification_dropout,
            "label_smoothing": config.label_smoothing,
            "label2id": getattr(config, 'label2id', {}),
            "id2label": getattr(config, 'id2label', {}),
        },
    }


def main():
    config = parse_args()
    now = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

    llm = scalarlm.SupermassiveIntelligence()

    dataset = get_dataset(config)

    train_args = build_train_args(config)
    
    status = llm.train(
        dataset,
        train_args=train_args
    )

    print(status)

    # `now` has one-second resolution, so disambiguate same-second submissions.
    save_dir = f'scalarlm_training/runs/{now}'
    suffix = 1
    while os.path.exists(save_dir):
        suffix += 1
        save_dir = f'scalarlm_training/runs/{now}-{suffix}'
    os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'train_status.json'), 'w+') as f:
        json.dump(status, f)

    print(f"Run status written to: {save_dir}/train_status.json")


if __name__ == '__main__':
    main()
