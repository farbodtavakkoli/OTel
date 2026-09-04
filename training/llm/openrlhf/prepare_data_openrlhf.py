"""Convert chat JSONL into the flat prompt/answer JSONL OpenRLHF's PPO/GRPO path expects (see readme_openrlhf.md)."""

import argparse
import json
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="chat JSONL -> OpenRLHF prompt/answer JSONL")
    parser.add_argument("--input", default="data/OTel_LLM_sample_10.jsonl", help="source chat JSONL")
    parser.add_argument("--output", default="data/otel_rl.jsonl", help="destination JSONL with prompt/answer")
    return parser.parse_args()


def convert(input_path, output_path):
    """Split each record at the final assistant turn: prior turns become prompt, its content becomes answer."""
    written = 0
    with open(input_path, "r", encoding="utf-8") as src, \
            open(output_path, "w", encoding="utf-8") as dst:
        for index, line in enumerate(src):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages")

            if not messages:
                logger.warning("row %d: no 'messages' field, skipping", index)
                continue
            if messages[-1].get("role") != "assistant":
                logger.warning("row %d: last turn is not 'assistant', skipping", index)
                continue

            prompt = messages[:-1]
            answer = messages[-1].get("content", "")
            if not prompt or not answer:
                logger.warning("row %d: empty prompt or empty answer, skipping", index)
                continue

            dst.write(json.dumps({"prompt": prompt, "answer": answer}, ensure_ascii=False) + "\n")
            written += 1
    return written


def main():
    args = parse_args()
    written = convert(args.input, args.output)
    if not written:
        raise SystemExit(f"no usable rows found in {args.input}")
    logger.info("wrote %d rows -> %s", written, args.output)


if __name__ == "__main__":
    main()
