# `inference/tenstorrent/llm`: LLM chat served by tt-inference-server

Talks to any chat model served by tt-inference-server through the OpenAI compatible
`POST /v1/chat/completions` endpoint. We validated it with `google/gemma-4-31B-it` on
Blackhole Galaxy, which is the default `--model` and the subject of the results below; pass
`--model` for any other served model. `infer_llm_tenstorrent.py` sends a chat prompt or a prompted
single label classification and prints the reasoning and the answer separately, with finish
reason, latency, decode rate and token usage.

**Hardware:** Tenstorrent Blackhole Galaxy, 32 Blackhole chips (verified with Gemma 4 31B).
Other devices in the tt-inference-server catalog are not verified by us.

## Files

- `infer_llm_tenstorrent.py`: chat and prompted classification client.

## Setup

Start a server and create the client venv as described in [`../README.md`](../README.md),
then, from this folder:

```bash
source ../.env_tenstorrent/bin/activate
ln -sf ../../../dev.env dev.env
python infer_llm_tenstorrent.py --model google/gemma-4-31B-it --no_thinking --max_tokens 256
```

## Client

```bash
python infer_llm_tenstorrent.py
python infer_llm_tenstorrent.py --no_thinking --seed 42 --max_tokens 256
python infer_llm_tenstorrent.py --stream --max_tokens 1024 --prompt "In one sentence, what does the 3GPP N2 interface connect?"
python infer_llm_tenstorrent.py --classify "My phone has had no signal since this morning" --no_thinking
```

Raw equivalent:

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"google/gemma-4-31B-it","messages":[{"role":"user","content":"What is the F1 interface?"}],"max_tokens":512,"chat_template_kwargs":{"enable_thinking":false}}'
```

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `localhost` | Server host |
| `--port` | `8000` | Server port |
| `--model` | `google/gemma-4-31B-it` | Served model name |
| `--prompt` | F1 interface question | User prompt |
| `--system_prompt` | `None` | Optional system message |
| `--classify` | `None` | Text to classify into one label; forces `temperature=0` |
| `--labels` | `billing,network_outage,device_setup,other` | Label set used by `--classify` |
| `--max_tokens` | `2048` | Token budget, reasoning included |
| `--temperature` | `0.7` | Sampling temperature |
| `--top_p` | `0.95` | Nucleus sampling top p |
| `--seed` | `None` | Per request seed; the same seed reproduces the output |
| `--no_thinking` | off | Sends `enable_thinking: false` |
| `--stream` | off | Streams over SSE and reports time to first token |
| `--timeout` | `600.0` | HTTP timeout in seconds |

Output:

```
endpoint      : http://localhost:8000/v1/chat/completions
model         : google/gemma-4-31B-it
thinking      : off
finish_reason : stop
latency       : 3.22s
decode rate   : 25.2 tok/s (81 completion tokens)
usage         : {'prompt_tokens': 31, 'total_tokens': 112, 'completion_tokens': 81, 'prompt_tokens_details': None}

--- response ---
The O-RAN F1 interface is a standardized logical interface used to connect ...
```

With thinking on, a `--- reasoning ---` block comes first. If `max_tokens` runs out during
reasoning, the client says so instead of printing an empty answer.

## Results

`google/gemma-4-31B-it`, measured 2026-10-21 on one Blackhole Galaxy, tt-inference-server v0.25.0 (tt-metal
`978f93b`, vllm-tt-plugin `a723030`, vLLM 0.26.0), 4 engines x 8 chips, prefix caching on but
defeated with `cache_salt`, thinking off. Each request had a unique prompt and
`cache_salt`, an exact input length including the chat template, and a fixed output length
through `ignore_eos`. Output tok/s counts generated tokens over wall time; total tok/s adds
the prompt tokens; per user tok/s is decode speed after the first token. Mean of 3 runs,
spread under 0.5%, zero errors.

| Point | Result |
|---|---|
| 128 in, 128 out, 1 user | TTFT 0.17 s, 26.3 tok/s per user |
| 128 in, 128 out, 128 users | TTFT 2.9 s, 19.4 tok/s per user, 1,695 output tok/s |
| 128 in, 1,024 out, 128 users | 2,481 output tok/s (2,791 total), 20.5 tok/s per user |
| 2K prompt, 128 out, 1 user | TTFT 0.60 s, 25.2 tok/s per user |
| 8K prompt, 128 out, 1 user | TTFT 2.54 s, 24.2 tok/s per user |
| 32K prompt, 128 out, 1 user | TTFT 11.06 s, 20.7 tok/s per user |
| 250K prompt, 1 user | TTFT 236 s |

Concurrency sweep, output tok/s (total tok/s in brackets):

| Users | 128 in, 1,024 out | 1,024 in, 1,024 out | TTFT mean (128 / 1,024 in) |
|---|---|---|---|
| 1 | 25.8 (29.0) | 25.2 (50.4) | 0.16 / 0.35 s |
| 8 | 197 (222) | 192 (384) | 0.27 / 0.53 s |
| 32 | 749 (843) | 714 (1,428) | 0.91 / 2.07 s |
| 64 | 1,326 (1,492) | 1,216 (2,431) | 1.63 / 4.34 s |
| 128 | 2,481 (2,791) | 2,155 (4,310) | 2.76 / 6.52 s |
| 256 | 2,490 (2,801) | 2,164 (4,328) | 41.8 / 52.1 s |

The server runs at most 128 requests at once; above that, requests queue, throughput stays
flat and TTFT grows.

### Other catalog models

Measured 2026-10-23 on the same Galaxy and image, each model alone on all 32 chips in the
shipped 4 engine config, 128 in and 1,024 out, mean of 3 runs. Gemma 4 12B and Qwen3 32B
used `ignore_eos` with thinking off; gpt-oss 120B used a long answer prompt without
`ignore_eos`, and every request still produced 1,024 tokens.

| Model | 1 user |
|---|---|
| `google/gemma-4-12B-it` | 46.0 tok/s, TTFT 0.13 s |
| `Qwen/Qwen3-32B` | 31.9 tok/s, TTFT 0.09 s |
| `openai/gpt-oss-120b` | 19.3 tok/s, TTFT 0.83 s |

gpt-oss 120B runs 4 engines of 32 sequences (128 in total); output was correct up to 32
concurrent requests and about half garbled at 64. Do not send `ignore_eos`: the server generates past `<|return|>` and its
harmony parser then fails the request. Thinking cannot be turned off, and JSON schema output
was cut off mid string in our check.

## Notes

- **Budget `max_tokens` for reasoning.** At 60 tokens with thinking on, `content` is often
  `null`.
- **Do not send `min_tokens` when benchmarking.** It halved decode speed (14 vs 26 tok/s per
  user); `ignore_eos` alone fixes the length, except on gpt-oss 120B (see above).
- **Use chat completions.** `/v1/completions` does not add `<bos>` and raw text prompts
  degrade into repetition.
- **Seeds reproduce under load.** The same seed gave identical text when sent alone and
  alongside 64 other requests.
