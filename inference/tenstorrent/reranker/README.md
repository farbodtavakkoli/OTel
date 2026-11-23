# `inference/tenstorrent/reranker`: Nemotron reranker served on Tenstorrent Blackhole

Talks to the Tenstorrent rerank server through `POST /v1/rerank`: a query and a list of documents
in, the documents ranked best first out, each with a relevance score in `[0, 1]` and the raw
logit. We validated it with `nvidia/llama-nemotron-rerank-1b-v2` on Blackhole Galaxy, on one chip
and on all 32. `rerank_tenstorrent.py` scores the same built-in query and documents as
[`inference/transformers/reranker`](../../transformers/reranker/README.md), prints the ranking,
checks that the relevant documents rank first, and writes a JSON reference artifact.

**Hardware:** Tenstorrent Blackhole Galaxy, 32 Blackhole chips (firmware bundle 19.12.0, KMD
2.7.0 on our test system), verified on 1 and on 32 chips. One server process drives any number of chips, one worker
per chip; requests go to the worker with the least outstanding work.

## Files

- `rerank_tenstorrent.py`: rerank client with a built-in relevance check.

## Setup: server

The image is public, about 8 GB compressed, and needs no login or Hugging Face token. It
downloads the model (2.4 GB, pinned revision) into the cache folder on its first start.

| Component | Version |
|---|---|
| Image | `ghcr.io/tenstorrent/tt-inference-server/tt-media-inference-server-nemotron-rerank:0.26.1-9228b26d5fc4` |
| Digest | `sha256:a18e9a71f1eb55e46be9575413863d6a6252e639616720a42e3170c41c87499d` |
| Model revision | `828765652b05bd439c9789d2a6d093db1caa1443` |

Install the host software as in [`../README.md`](../README.md), then make a cache folder the
server (user 1000) can write to, and list the chips to use by PCI address. All 32 chips of a
Galaxy:

```bash
mkdir -p $HOME/rerank-volume && chmod a+rwx $HOME/rerank-volume
RR_CHIPS=$(for d in /sys/class/tenstorrent/tenstorrent!*; do basename $(readlink -f $d/device); done | sort | paste -sd,)
```

For one chip, pass that chip's address instead (`RR_CHIPS=0000:46:00.0`) and its device
(`--device /dev/tenstorrent/N`, where N is the number in front of the address in the loop above).

Start the server:

```bash
IMG=ghcr.io/tenstorrent/tt-inference-server/tt-media-inference-server-nemotron-rerank:0.26.1-9228b26d5fc4
M=/home/container_app_user/tt-metal
S=$M/models/demos/blackhole/llama_nemotron_rerank_1b/serve
docker run -d --name nemo --user 1000:1000 \
  --device /dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --mount type=bind,src=$HOME/rerank-volume,dst=/cache \
  -p 127.0.0.1:8000:8000 \
  -e RR_CHIPS=$RR_CHIPS -e RR_PORT=8000 \
  -e TT_METAL_TDP_LIMIT_WATTS=160 \
  -e HOME=/cache/home -e TT_CACHE_PATH=/cache/weights \
  -e TT_METAL_CACHE=/cache/tt-metal -e HF_HOME=/cache/hf \
  -e TOKENIZERS_PARALLELISM=false -e TT_MM_THROTTLE_PERF=0 -e TT_LOGGER_LEVEL=ERROR \
  -e ARCH_NAME=blackhole -e TT_METAL_HOME=$M \
  -e NEMOTRON_RERANK_REVISION=828765652b05bd439c9789d2a6d093db1caa1443 \
  -e PYTHONPATH=$S:$M/ttnn:$M:$M/tools -e LD_LIBRARY_PATH=$M/build/lib \
  --workdir $S --entrypoint bash $IMG \
  -c "source $M/python_env/bin/activate && bash entrypoint_selfcontained.sh"
```

The image starts a different server by default; the command above swaps in the reranker. The
first start downloads the model and compiles the kernels (110 s of warm-up on one chip in our
run); later starts reuse the cache folder. The server is ready when `/health` reports
`"status":"ok"` with every worker `ok`:

```bash
curl -s localhost:8000/health | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['status'], sum(w['state'] == 'ok' for w in d['workers']), 'of', len(d['workers']), 'workers ok')"
```

Stop it with `docker stop -t 120 nemo && docker rm nemo`, then reset the chips with
`tt-smi -r all` before the next server, as in [`../README.md`](../README.md).

## Client

```bash
source ../.env_tenstorrent/bin/activate
ln -sf ../../../dev.env dev.env
python rerank_tenstorrent.py --output $OUTPUT_DIR/inference_reranker_tenstorrent/reference_rerank.json
python rerank_tenstorrent.py --query "How do I reset my router?" --documents_file my_docs.json --top_n 5
```

Raw equivalent:

```bash
curl -s localhost:8000/v1/rerank -H 'Content-Type: application/json' -d '{
  "model": "nvidia/llama-nemotron-rerank-1b-v2",
  "query": "Which inference engines support AMD ROCm?",
  "documents": ["vLLM supports AMD ROCm.", "PostgreSQL is a relational database."],
  "top_n": 2, "return_documents": true}'
```

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `localhost` | Server host |
| `--port` | `8000` | Server port |
| `--model` | `nvidia/llama-nemotron-rerank-1b-v2` | Served model name |
| `--query` | built-in | Query to rerank against |
| `--documents_file` | `None` | JSON/JSONL/text documents (omit for the built-in set) |
| `--top_n` | all | Return only the best N documents |
| `--output` | `None` | Path for the JSON reference artifact |
| `--timeout` | `120.0` | HTTP timeout in seconds |

Output on one Blackhole Galaxy, all 32 chips:

```text
endpoint      : http://localhost:8000/v1/rerank
model         : nvidia/llama-nemotron-rerank-1b-v2
latency       : 11.8 ms (4 pairs)
usage         : {'total_tokens': 95}

Q: Which inference engines support AMD ROCm?
  #1 score=0.998866 logit=+6.7807  vLLM supports AMD ROCm.
  #2 score=0.051690 logit=-2.9094  SGLang ships a ROCm build for AMD Instinct accelerators.
  #3 score=0.000000 logit=-24.7134  The Eiffel Tower is located in Paris, France.
  #4 score=0.000000 logit=-26.3390  PostgreSQL is a relational database.

sanity: min(relevant)=-2.9094 > max(irrelevant)=-24.7134 -> PASS
sanity: top-2 == relevant set -> PASS

SANITY: PASS
```

The sanity check compares logits: the relevance score saturates at 0 for clearly irrelevant
documents. The script exits non-zero if the check fails.

## Results

Measured 2026-11-23 on one Blackhole Galaxy with the image above. Each request carried a 12-word
query and 25 documents of 60 words; closed loop, each in-flight request sent back to back for
20 s, a new connection per request, zero errors at every point in the table.

| Chips | Requests in flight | p50 latency | Pairs per second |
|---|---|---|---|
| 1 | 1 | 44 ms | 561 |
| 1 | 8 | 222 ms | 896 |
| 1 | 32 | 893 ms | 894 |
| 32 | 1 | 44 ms | 567 |
| 32 | 32 | 49 ms | 15,527 |
| 32 | 64 | 100 ms | 15,506 |
| 32 | 128 | 200 ms | 15,594 |

One chip saturates at 8 requests in flight. All 32 chips keep the one-chip latency up to 32
requests in flight and reach about 15,500 pairs per second, 17 times one chip.

Tenstorrent's validation of this image, on one chip: Pearson 0.9996 against the Hugging Face fp32
model on 1,000 MIRACL-en pairs, nDCG@10 0.6833 against 0.6840.

## Notes

- **Limits:** up to 1,000 documents per request; each query and document pair is cut at 512
  tokens.
- **Keep at most about 128 requests in flight** when each request opens a new connection: at 256,
  some connections were reset in our runs. Reuse connections or add a client-side limit.
- **`relevance_score` is the sigmoid of `logit`.** Rank on either; use the logit to compare
  clearly irrelevant documents, where the score rounds to 0.
- **Reset with `tt-smi -r all` after every stop**, as for the LLM server.
