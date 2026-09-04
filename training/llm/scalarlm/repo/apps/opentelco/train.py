import os
import json
import argparse
import scalarlm
import datetime
import yaml


# Training config
# Make sure to at least modify one setting in the config before submitting a new job otherwise the system would recognize your
# job as already included. You can simply modify the max_steps to a different number and submit a job
def get_args():
    parser = argparse.ArgumentParser()

    # Sample data for training
    # Make sure your input data for training follows the same pattern
    # Otherwise you would need to modify "format_chat_template" function
    parser.add_argument('--data_path', default = 'apps/opentelco/data/test_data.json')

    # In case you want to include reasoning in the output during training.
    # Adding reasoning tokens will increase the memory usage during training
    # which means you may need to decrease the number of batch size
    parser.add_argument('--reasoning_mode', default = True)

    # Training hyperparameters
    # Set max_token_block_size to be as big as the longest sequence of your training set.
    # consider using max_token_block_size of 4000 or less. For any values larger than 4000, you may run out of memory
    parser.add_argument('--max_steps', default = 5)
    parser.add_argument('--learning_rate', default = 0.005)
    parser.add_argument('--max_token_block_size', default = 30)
    parser.add_argument('--steps_per_checkpoint', default = 100)

    # LoRA Configuration
    parser.add_argument('--r', default = 8)
    parser.add_argument('--lora_alpha', default = 16)
    parser.add_argument('--lora_dropout', default = 0.05)
    parser.add_argument('--target_modules', default = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                                            choices=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])

    # Do not need to be updated
    parser.add_argument('--batch_size', default = 2)
    parser.add_argument('--new_data', default=False)
    parser.add_argument('--use_lora', default=True)
    parser.add_argument('--use_tokenformer', default=False)
    parser.add_argument('--training_mode', default = 'language_model', choices=['language_model', 'embedding'])

    return parser.parse_args()

# Data pre-processing and adding special tokens before training
# The special tokens of Qwen models are added below for your convenience
def format_chat_template(example, reasoning_mode = True, scalarlm_format = False):
    if reasoning_mode:
        formatted_text = f"""<im_start>user
{example["Input_Prompt"]}<im_end>

<im_start>model
Root Cause Reasoning Trace:
{example["Reasoning_Trace"]}

Most Likely Root Cause:
{example['Generated_Label']}<im_end>"""
    else:
        formatted_text = f"""<im_start>user
        {example["Input_Prompt"]}"<im_end>
        <im_start>model
        The root cause is \n\n\\boxed{example["Gold_Label"]}<im_end>"""

    if scalarlm_format:
        out = {'input': '', 'output': formatted_text}
    else:
        out = {"text": formatted_text}

    return out

# Loading data
def get_dataset(data_path, reasoning_mode):
    dataset = []

    with open(data_path, 'r+') as f:
        data = json.load(f)

    for example in data:
        if config.new_data:
            dataset.append(data['record']) # Pre formatted
        else:
            dataset.append(format_chat_template(example, reasoning_mode = reasoning_mode, scalarlm_format = True))

    return dataset

if __name__ == '__main__':
    config = get_args()
    now = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

    llm = scalarlm.SupermassiveIntelligence()

    dataset = get_dataset(config.data_path, config.reasoning_mode)

    train_args = {
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "max_token_block_size": config.max_token_block_size,
        "steps_per_checkpoint": config.steps_per_checkpoint,
        "training_mode": config.training_mode,
        "adapter_type": "lora" if config.use_lora else "tokenformer" if config.use_tokenformer else None,
        "lora_config": {
            "r": config.r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "target_modules": config.target_modules
        } if config.use_lora else None
    }


    status = llm.train(
        dataset,
        train_args=train_args
    )

    print(status)

    # Saving training history locally
    save_dir = f'scalarlm_training/runs/{now}'
    os.makedirs(save_dir, exist_ok = False)

    with open(os.path.join(save_dir, 'train_status.json'), 'w+') as f:
        json.dump(status, f)
