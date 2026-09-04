import os
import json
import time
import traceback
import scalarlm
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # Sample data for training
    # Make sure your input data for training follows the same pattern
    # Otherwise you would need to modify "get_dataset" function
    parser.add_argument('--test_data_path', default = 'apps/opentelco/data/test_data.json')

    # Once you submit a job for training, the code automatically create a local folder to keep logs of submitted jobs for you.
    # In order to save the results of inference in the same folder, you need to modify this variable to match with the training job folder.
    parser.add_argument('--run_folder', default = 'scalarlm_training/runs/2025_12_15__15')

    # Use your model_id to inference from your fine tuned model.
    # You receive the model_id immidiately after submitting a job.
    # You also can find your model_id in the "scalarlm_training" folder
    parser.add_argument('--model_name', default = 'dfff7535905affe768729e8b82971743ffda1a0f73de405c45842fcd1d46b15e')

    # The max tokens allowed during inference for the current deployments is 4096.
    # Please keep in mind the input field for TeleLogs itself has ~2900 tokens which means the max_token field can only handle up to 1200 tokens.
    # If you do not need all the 1200 tokens during inference, only pass the number of tokens you need.
    # Having a larger max_token than your longest sequence would simply force
    # every sequence during inference to be right padded which is not memory efficient.
    parser.add_argument('--max_tokens', default = 500)

    # Depending on your max_token, you can inference from different number of sequence in a batch.
    # You may run out of memory if you increase the batch size significatly while having a large max_tokens
    parser.add_argument('--batch_size', default = 2)

    # Do not need to be updated
    parser.add_argument('--inference_mode', default = 'language_model',
                                             choices=['language_model', 'embedding'])

    return parser.parse_args()

# Data pre-processing and adding special tokens before training
# The special tokens of Qwen models are added below for your convenience
def get_dataset(config):

    dataset = []
    with open(config.test_data_path, 'r+') as f:
        test_data = json.load(f)

    count = len(test_data)

    for i in range(count):
        dataset.append(
            f"""<im_start>user
{test_data[i]['Input_Prompt']}<im_end>

<im_start>model"""
            )

    return dataset, test_data


if __name__ == '__main__':
    config = get_args()

    llm = scalarlm.SupermassiveIntelligence()

    dataset, test_data = get_dataset(config)
    results = []
    processed_indices = []
    batch_size = int(config.batch_size)

    # Running inference on the data in batches and logging the output incrementally
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i: i+batch_size]
        try:
            out = llm.generate(
                    prompts=batch,
                    model_name = config.model_name,
                    max_tokens = config.max_tokens
            )
            results += out
            processed_indices += list(range(i, min(i+batch_size, len(dataset))))
            remaining_instances = max(0, len(dataset) - (i + batch_size))
            print(f'Completed an inference for a batch of size {batch_size}. There are  {remaining_instances} instances left to inference.')
        except Exception as e:
            print('batch ', i, ' failed')
            print(f'Error type: {type(e).__name__}')
            print(f'Error message: {e}')
            traceback.print_exc()
            continue

    output = []
    for idx, result_idx in enumerate(processed_indices):
        output.append({
            'ground_truth_answer': test_data[result_idx]['Gold_Label'],
            'input_question': dataset[result_idx],
            'generated_response': results[idx]
        })

    try:
        with open(os.path.join(config.run_folder, 'results.json'), 'w+') as f:
            json.dump(output, f, indent=4)

    except (FileNotFoundError, OSError):
        # If the folder doesn't exist, save in the current directory
        local_filename = f"results_{config.model_name}.json"
        print(f"Folder '{config.run_folder}' not found. Saving locally as {local_filename}")
        with open(local_filename, 'w+') as f:
            json.dump(output, f, indent=4)
