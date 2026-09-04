import scalarlm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--api-url", default="http://localhost:8000")
args = parser.parse_args()

scalarlm.api_url = args.api_url

def get_dataset():
    dataset = []
    count = 1
    for i in range(count):
        dataset.append({"input": f"What is {i} + {i}?", "output": str(i + i)})
    return dataset

llm = scalarlm.SupermassiveIntelligence()
dataset = get_dataset()
status = llm.train(
    dataset,
    train_args={
        "max_steps": 10,
        "learning_rate": 3e-3,
        "gpus": 1,
        "nodes": 1,
        "max_token_block_size": 4096,
        "adapter_type": "lora",
        "steps_per_checkpoint": 10000
    },
)
print(status)