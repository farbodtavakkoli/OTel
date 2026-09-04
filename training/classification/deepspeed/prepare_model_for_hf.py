"""Rename the trained checkpoint's custom-head weights for plain HF loading — see readme_classification.md."""
import argparse
import torch
import shutil
from pathlib import Path
from safetensors.torch import load_file, save_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description="Prepare a trained classifier checkpoint for HF upload")
    parser.add_argument("--source", type=str, default="best_model", help="Trained checkpoint dir")
    parser.add_argument("--output", type=str, default="best_model_hf_ready", help="Output dir for the HF-ready model")
    return parser.parse_args()


def prepare_model(source, output):
    """Rename score.1.weight to score.weight, drop dropout params, copy configs, and verify loading."""
    print(f"Loading model from: {source}")

    weights_path = Path(source) / "model.safetensors"
    state_dict = load_file(weights_path)
    
    # Check current keys
    print("\nOriginal classification head keys:")
    for key in state_dict.keys():
        if "score" in key:
            print(f"  {key}: {state_dict[key].shape}")
    
    # Rename score.1.weight -> score.weight for standard loading
    new_state_dict = {}
    for key, value in state_dict.items():
        if key == "score.1.weight":
            new_key = "score.weight"
            print(f"\nRenaming: {key} -> {new_key}")
            new_state_dict[new_key] = value
        elif key == "score.0.weight" or key == "score.0.bias":
            print(f"Skipping: {key} (dropout layer)")
            continue
        else:
            new_state_dict[key] = value
    
    print("\nNew classification head keys:")
    for key in new_state_dict.keys():
        if "score" in key:
            print(f"  {key}: {new_state_dict[key].shape}")
    
    output_path = Path(output)
    output_path.mkdir(exist_ok=True)

    new_weights_path = output_path / "model.safetensors"
    save_file(new_state_dict, new_weights_path)
    print(f"\nSaved weights to: {new_weights_path}")
    
    # Copy other files (config, tokenizer, etc.)
    files_to_copy = [
        "config.json",
        "tokenizer.json", 
        "tokenizer_config.json",
        "chat_template.jinja",
        "training_args.bin"
    ]
    
    for filename in files_to_copy:
        src = Path(source) / filename
        if src.exists():
            shutil.copy(src, output_path / filename)
            print(f"Copied: {filename}")
    
    # Verify the model loads correctly
    print("\n" + "="*50)
    print("Verifying model loads correctly...")
    print("="*50)
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            output,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(output)
        
        # Check the score layer
        print(f"\nModel score layer type: {type(model.score)}")
        print(f"Model score layer shape: {model.score.weight.shape}")
        print(f"Number of labels: {model.config.num_labels}")
        print(f"Label mapping: {model.config.label2id}")
        
        print("\nModel loads correctly with standard from_pretrained()!")
        print(f"\nReady for upload: {output}/")

    except Exception as e:
        print(f"\nError loading model: {e}")
        raise


def main():
    args = parse_args()
    prepare_model(args.source, args.output)


if __name__ == "__main__":
    main()
