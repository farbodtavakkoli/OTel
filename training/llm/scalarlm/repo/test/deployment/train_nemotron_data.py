import scalarlm
from datasets import load_dataset

scalarlm.api_url = "https://qwen4b-api.scalarllm.com"

# ChatML tokens used by Qwen models
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

def format_messages(messages):
    """Convert a list of {role, content} messages into ChatML input/output pairs.

    Everything up to (and including) the last user turn becomes the input.
    The final assistant turn becomes the output.
    """
    # Find the last assistant message index
    last_asst_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            last_asst_idx = i
            break

    if last_asst_idx is None:
        return None  # No assistant reply to train on

    # Build input: all messages before the last assistant turn + the assistant header
    input_parts = []
    for msg in messages[:last_asst_idx]:
        role = msg["role"]
        content = msg.get("content") or ""
        input_parts.append(f"{IM_START}{role}\n{content}{IM_END}")

    # Append the assistant header so the model learns to continue from here
    input_parts.append(f"{IM_START}assistant\n")
    input_text = "\n".join(input_parts)

    # Build output: the assistant response + end token
    output_text = f"{messages[last_asst_idx]['content']}{IM_END}"

    return {"input": input_text, "output": output_text}


def get_dataset(num_samples=2048):
    ds = load_dataset(
        "nvidia/Nemotron-Instruction-Following-Chat-v1",
        split="chat_if",
        streaming=True,
    )

    dataset = []
    for i, row in enumerate(ds):
        if i >= num_samples:
            break
        result = format_messages(row["messages"])
        if result:
            dataset.append(result)

    print(f"Loaded {len(dataset)} examples")
    return dataset


llm = scalarlm.SupermassiveIntelligence()
dataset = get_dataset(num_samples=2048)

status = llm.train(
    dataset,
    train_args={
        "max_steps": 2048,
        "learning_rate": 3e-4,
        "gpus": 1,
        "nodes": 1,
        "max_token_block_size": 4096,
        "steps_per_checkpoint": 10000,
    },
)
print(status)
