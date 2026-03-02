import os
import json
import time
import scalarlm
import argparse
import traceback


def get_args():

    parser = argparse.ArgumentParser()

    # Sample data for inference
    parser.add_argument('--test_data_path', default = 'data/test_data.json')

    # Once you submit a job for training, the code automatically create a local folder to keep logs of submitted jobs for you.
    # In order to save the results of inference in the same folder, you need to modify this variable to match with the training job folder.
    parser.add_argument('--run_folder', default = 'scalarlm_training/runs/2026_02_26__05_06_12')

    # Use your model_id to inference from your fine tuned model. 
    # You receive the model_id immidiately after submitting a job. 
    # You also can find your model_id in the "scalarlm_training" folder
    # To inference from the model without adapter, remove or comment "model_name = config.model_name" from "llm.generate"
    parser.add_argument('--model_name', default = '9f0122ddc2e55a14f1ae195c871c736b0673d98d2f0cfcf77b85a5863b98b052')
     
    # The max tokens allowed during inference
    # max tokens is total input plus output tokens allowed during the inference. Example:
    # Deployed 4096
    # Input 3000
    # max_token 1096
    parser.add_argument('--max_tokens', default = 500)

    # Depending on your max_token, you can inference from different number of sequence in a batch.
    parser.add_argument('--batch_size', default = 2)

    # Inference mode depending on wheather you want to inference from language model or embedding model
    # There are models (usually embedding models) that vLLM does not support inference from. 
    # In this case, you can inference from the base model and use sentencetransformer to inference from embedding locally
    parser.add_argument('--inference_mode', default = 'language_model', 
                                             choices=['language_model', 'embedding'])
    
    # Model type for special tokens formatting (gemma3, qwen3, llama3, rnj-1, olmo3, mistral, gpt-oss_reasoning, gpt_oss_it)
    parser.add_argument('--model_type', default='gemma3')
    
    return parser.parse_args()


# Special tokens formatting for different model types
# For inference, completion is left empty and the string ends at the assistant/model turn.
def format_conversation_inference(example, model_type="gemma3", max_prompt_chars=1000):
    prompt = str(example.get('prompt', '')) or ""
    prompt = prompt[:max_prompt_chars]

    if model_type.lower() == "qwen3" or ("qwen" in model_type.lower()):
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    elif model_type.lower() == "llama3":
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    elif model_type.lower() == "gemma3" or ("gemma" in model_type.lower()):
        # Matches your desired inference shape (assistant/model turn left open)
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n\n\n<start_of_turn>model"

    elif model_type.lower() == "rnj-1" or ("rnj" in model_type.lower()):
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{prompt}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )

    elif model_type.lower() == "olmo3" or ("olmo" in model_type.lower()):
        return f"<|endoftext|><|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    elif model_type.lower() == "mistral":
        # Leaves the assistant generation empty (no closing </s> yet)
        return f"<s>[SYSTEM_PROMPT][/SYSTEM_PROMPT][INST]{prompt}[/INST]"

    elif model_type.lower() == "gpt-oss_reasoning":
        reasoning = str(example.get('reasoning', '')) or ""
        return (
            "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
            "Knowledge cutoff: 2024-06\nCurrent date: 2026-02-11\n\n"
            "Reasoning: low\n\n"
            "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
            "<|start|>developer<|message|># Instructions\n\n"
            "reasoning_language: English\n\n"
            "You are an Open Source Telecom Question answering model that uses provided contexts to answer telecom questions.\n\n"
            "<|end|>"
            f"<|start|>user<|message|>{prompt}<|end|>"
            "<|start|>assistant<|channel|>analysis<|message|>"
            f"{reasoning}"
            "<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
        )

    elif model_type.lower() == "gpt_oss_it":
        return (
            "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
            "Knowledge cutoff: 2024-06\nCurrent date: 2026-02-11\n\n"
            "Reasoning: low\n\n"
            "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
            f"<|start|>user<|message|>{prompt}<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
        )

    else:
        return f"User: {prompt}\nAssistant:"

def get_dataset(config):
    """Load dataset for inference.
    
    For embedding mode: Returns raw sentences to encode
    For language_model mode: Returns formatted prompts with chat templates
    """
    dataset = []
    with open(config.test_data_path, 'r+') as f:
        test_data = json.load(f)
    
    count = len(test_data)

    if config.inference_mode == 'embedding':
        # For embedding models, pass raw text without special tokens
        for i in range(count):
            # If data has anchor format (embedding training data)
            if 'anchor' in test_data[i]:
                dataset.append(test_data[i]['anchor'])
            # If data has sentence1/sentence2 format (embedding training data)
            elif 'sentence1' in test_data[i]:
                dataset.append(test_data[i]['sentence1'])
            # If data has prompt format (language model data)
            elif 'prompt' in test_data[i]:
                dataset.append(test_data[i]['prompt'])
            # If data has Input_Prompt format (legacy format)
            elif 'Input_Prompt' in test_data[i]:
                dataset.append(test_data[i]['Input_Prompt'])
            else:
                # Fallback: try to find any text field
                print(f"Warning: Unexpected data format at index {i}: {test_data[i].keys()}")
    else:
        # For language model inference, use configurable chat template
        for i in range(count):
            # Extract prompt from various possible field names
            if 'prompt' in test_data[i]:
                prompt_text = test_data[i]['prompt']
            elif 'Input_Prompt' in test_data[i]:
                prompt_text = test_data[i]['Input_Prompt']
            elif 'anchor' in test_data[i]:
                prompt_text = test_data[i]['anchor']
            else:
                print(f"Warning: No prompt field found at index {i}: {test_data[i].keys()}")
                continue
            
            # Build example dict for format_conversation_inference
            example = {
                'prompt': prompt_text,
                'reasoning': test_data[i].get('reasoning', '')
            }
            
            dataset.append(
                format_conversation_inference(example, model_type=config.model_type)
            )

    return dataset, test_data


if __name__ == '__main__':
    config = get_args()

    llm = scalarlm.SupermassiveIntelligence()

    dataset, test_data = get_dataset(config)
    results = []
    processed_indices = []
    batch_size = int(config.batch_size)
    
    for i in range(0, 5, batch_size):
        batch = dataset[i: i+batch_size]
        try:
            t0 = time.time()

            # To inference from the model without adapter, remove or comment "model_name = config.model_name" from "llm.generate"
            out = llm.generate(
                    prompts=batch, 
                    model_name = config.model_name,
                    max_tokens = config.max_tokens
            )
            results += out
            processed_indices += list(range(i, min(i+batch_size, len(dataset))))
            t1 = time.time()
            print(f'took {t1-t0} seconds to do batch of size {batch_size}')
        except Exception as e:
            traceback.print_exc()
            print(e)
            print('batch ', i, ' failed')
            continue
    
    output = []
    for idx, result_idx in enumerate(processed_indices):
        # Extract ground truth from various possible field names
        gt_answer = (
            test_data[result_idx].get('Gold_Label') or 
            test_data[result_idx].get('completion') or 
            test_data[result_idx].get('positive') or 
            ''
        )
        output.append({
            'gt_answer': gt_answer,
            'question': dataset[result_idx],
            'response': results[idx]
        })
    
    with open(os.path.join(config.run_folder, 'results.json'), 'w+') as f:
        json.dump(output, f, indent = 4)
        
 