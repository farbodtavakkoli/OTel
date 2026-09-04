import scalarlm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--api-url", default="http://localhost:8000")
parser.add_argument("--model", default=None)
args = parser.parse_args()

scalarlm.api_url = args.api_url
llm = scalarlm.SupermassiveIntelligence()

dataset = ["Hello, how are you?", "What is 10 + 15?", "Tell me a story"]

results = llm.generate(
    model_name=args.model,
    prompts=dataset,
    max_tokens=32,
)
print(f"model:\n{args.model}\n\nresults:\n{results}")