import os
import json
import time
import datetime
import scalarlm
import argparse
import traceback


def get_args():

    parser = argparse.ArgumentParser()

    # Sample data for inference
    parser.add_argument('--test_data_path', default = 'data/classification_data.json')

    # Use your model_id to inference from your fine tuned model. 
    # To inference from the model without adapter, remove or comment "model_name = config.model_name" from "llm.generate"
    parser.add_argument('--model_name', default = '9f0122ddc2e55a14f1ae195c871c736b0673d98d2f0cfcf77b85a5863b98b052')
     
    # max tokens is total input plus output tokens allowed during the inference. Example:
    # 4096 (Deployed context length) = 3000 (Input tokens) + 1096 (max output tokens allowed)
    parser.add_argument('--max_tokens', default = 500)

    # Depending on your max_token, you can inference from different number of sequence in a batch.
    parser.add_argument('--batch_size', default = 2)

    # Inference mode depending on wheather you want to inference from language model or embedding model
    # There are models (usually embedding models) that vLLM does not support inference from. 
    # In this case, you can inference from the base model and use sentencetransformer to inference from embedding locally
    parser.add_argument('--inference_mode', default = 'language_model', choices=['language_model', 'embedding'])
    
    # Model type for special tokens formatting (gemma3, qwen3, llama3, rnj-1, olmo3, mistral, gpt-oss_reasoning, gpt_oss_it, lfm, phi4)
    parser.add_argument('--model_type', default='rnj-1')
    
    return parser.parse_args()


# Special tokens formatting for different model types
def format_conversation_inference(example, model_type):
    prompt = str(example.get('prompt', '')) 

    if model_type.lower() == "qwen3":
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    elif model_type.lower() == "llama3":
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
 
    elif model_type.lower() == "gemma3":
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n\n\n<start_of_turn>model"

    elif model_type.lower() == "rnj-1":
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

    elif model_type.lower() == "olmo3":
        return f"<|endoftext|><|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    elif model_type.lower() == "mistral":
        return f"<s>[SYSTEM_PROMPT][/SYSTEM_PROMPT][INST]{prompt}[/INST]"

    elif model_type.lower() == "lfm":
        return f"<|startoftext|><|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    elif model_type.lower() == "phi4":
        return f"<|user|>{prompt}<|end|><|assistant|>"

    elif model_type.lower() == "gpt-oss_reasoning":
        return f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2026-02-11\n\nReasoning: low\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions\n\nreasoning_language: English\n\nYou are an Open Source Telecom Question answering model that uses provided contexts to answer telecom questions.\n\n<|end|><|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>analysis<|message|>\n"

    elif model_type.lower() == "gpt_oss_it":
        return f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2026-02-11\n\nReasoning: low\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>final<|message|>"

    else:
        return f"User: {prompt}\nAssistant:"

def get_dataset(config):
    dataset = []
    with open(config.test_data_path, 'r+') as f:
        test_data = json.load(f)
    
    count = len(test_data)

    if config.inference_mode == 'embedding':
        # For embedding models, pass raw text without special tokens
        for i in range(count):
            if 'anchor' in test_data[i]:
                dataset.append(test_data[i]['anchor'])
            else:
                # Fallback: try to find any text field
                print(f"Warning: Unexpected data format at index {i}: {test_data[i].keys()}")

    elif config.inference_mode == 'language_model':
        # For language model inference, use configurable chat template
        for i in range(count):
            if 'prompt' in test_data[i]:
                prompt_text = test_data[i]['prompt']
            else:
                print(f"Warning: No prompt field found at index {i}: {test_data[i].keys()}")
                continue
            
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
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i: i+batch_size]
        try:
            t0 = time.time()

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
            print('batch ', i, ' failed')
            continue
    
    output = []
    for idx, result_idx in enumerate(processed_indices):
        gt_answer = test_data[result_idx].get('completion', '')
        output.append({
            'ground_truth_response': gt_answer,
            'input_prompt': dataset[result_idx],
            'generated_response': results[idx]
        })

    now = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
 
    save_dir = f'scalarlm_inference/runs/{now}'
    os.makedirs(save_dir, exist_ok = False)

    print(f"Saving inference results to {save_dir}/inference_results.json")
    
    with open(os.path.join(save_dir, 'inference_results.json'), 'w+') as f:
        json.dump(output, f, indent = 4)
        
