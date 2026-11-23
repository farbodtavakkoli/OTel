# `inference/tenstorrent/embedding`: EmbeddingGemma-2 served on Tenstorrent Blackhole Galaxy

Talks to the Tenstorrent embedding server through the OpenAI compatible `POST /v1/embeddings`
endpoint. We validated it with `google/embeddinggemma-2` on Blackhole Galaxy, data parallel over
all 32 chips. `embed_tenstorrent.py` encodes the same built-in queries and documents as
[`inference/transformers/embedding`](../../transformers/embedding/README.md) with EmbeddingGemma's
query and document prompts, prints the ranked cosine similarity matrix, checks that the relevant
documents rank first, and writes a JSON reference artifact.

**Hardware:** Tenstorrent Blackhole Galaxy, 32 Blackhole chips (firmware bundle 19.12.0, KMD
2.7.0 on our test system), verified with one model copy per chip (DP=32).

## Files

- `embed_tenstorrent.py`: embedding client with a built-in relevance check.

## Setup: server

| Component | Version |
|---|---|
| Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.26.1-51738c0c-embeddinggemma2` (public) |
| Digest | `sha256:ef283d2cbd68b69887530bb8eb8b7634ffa22ffb1ca962be6b2accc7b3486faa` |
| tt-metal | `51738c0c75f9c1a36f4a568152b194f870030b53` |
| Model revision | `914f7f89142e33e77833254d9c9b90c3cef7303b` |

Install the host software as in [`../README.md`](../README.md). Download the pinned weights
(about 1.5 GB) into a work folder, with a Hugging Face read token (if the download is refused, accept the
model's license on its Hugging Face page), then let the server (user 1000) write there:

```bash
export WORK=$HOME/embeddinggemma && mkdir -p $WORK/hf
HF_HOME=$WORK/hf HF_TOKEN=hf_... hf download google/embeddinggemma-2 --revision 914f7f89142e33e77833254d9c9b90c3cef7303b
chmod -R a+rwx $WORK
```

Start the server on all 32 chips. It runs 32 model copies, one per chip, behind 16 front ends on
port 8132:

```bash
IMG=ghcr.io/tenstorrent/tt-media-inference-server:0.26.1-51738c0c-embeddinggemma2
docker run -d --name embgemma2-dp32 --network host --user 1000:1000 --shm-size 16g --cap-add SYS_NICE \
  --device /dev/tenstorrent -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v $WORK:/work -e EMB_ROOT=/work -e EMB_TT_METAL=/home/container_app_user/tt-metal -e TT_MM_THROTTLE_PERF=0 \
  --entrypoint bash $IMG \
  -c 'cd $TT_METAL_HOME/models/demos/embedding_gemma2/serving_study && DP=32 FRONTENDS=16 MAXB=32 bash run_fastapi_dp.sh; sleep infinity'
```

The first start converts the weights and compiles the kernels, which takes a few minutes; later starts reuse
`$WORK` and took 61 s in our run. The server is ready when all 16 front ends report it:

```bash
grep -l "Embedding pool ready" $WORK/logs/fastapi_dp32_k16_f*.log | wc -l   # 16 when ready
```

`--cap-add SYS_NICE` lets each worker bind its host buffers to the chip's NUMA node. Without it the kernel refuses
the binding with a `hwloc_set_area_membind(): failed` warning, and the launch script, which stops if front end 0's
log contains `failed`, never starts the other 15 front ends.

Stop it with `docker rm -f embgemma2-dp32`, then reset the chips with `tt-smi -r all` before the
next server, as in [`../README.md`](../README.md).

## Client

```bash
source ../.env_tenstorrent/bin/activate
ln -sf ../../../dev.env dev.env
python embed_tenstorrent.py --output $OUTPUT_DIR/inference_embedding_tenstorrent/reference_embedding.json
python embed_tenstorrent.py --queries_file my_queries.json --documents_file my_docs.json --truncate_dim 256
```

Raw equivalent (the prompt is part of the text):

```bash
curl -s http://localhost:8132/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model": "google/embeddinggemma-2", "input": ["task: search result | query: What GPU runtimes support ROCm?"]}'
```

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `localhost` | Server host |
| `--port` | `8132` | Server port |
| `--model` | `google/embeddinggemma-2` | Served model name |
| `--queries_file` | `None` | JSON/JSONL/text queries (omit for the built-in set) |
| `--documents_file` | `None` | JSON/JSONL/text documents (omit for the built-in set) |
| `--truncate_dim` | `None` | Matryoshka output dim (768/512/256/128), renormalized client side |
| `--no_prompts` | off | Send the texts without the query and document prompts |
| `--output` | `None` | Path for the JSON reference artifact |
| `--timeout` | `120.0` | HTTP timeout in seconds |

Output on one Blackhole Galaxy, all 32 chips:

```text
endpoint      : http://localhost:8132/v1/embeddings
model         : google/embeddinggemma-2
latency       : queries 12.3 ms (2 texts), documents 17.3 ms (4 texts)
dim           : 768
query[0] first8=[0.040114, 0.04992, 0.007117, 0.04453, 0.008444, 0.007402, -0.010108, 0.011343]

Q0: What GPU runtimes support ROCm?
  #1 cos=+0.808663  vLLM supports NVIDIA CUDA and AMD ROCm.
  #2 cos=+0.750734  SGLang provides a ROCm build for AMD Instinct accelerators.
  #3 cos=+0.597119  SQLite is an embedded database.
  #4 cos=+0.524741  The Eiffel Tower is located in Paris, France.
  sanity: min(relevant)=+0.750734 > max(irrelevant)=+0.597119 -> PASS

Q1: Which database is embedded and serverless?
  #1 cos=+0.794556  SQLite is an embedded database.
  #2 cos=+0.648514  vLLM supports NVIDIA CUDA and AMD ROCm.
  #3 cos=+0.629232  SGLang provides a ROCm build for AMD Instinct accelerators.
  #4 cos=+0.575772  The Eiffel Tower is located in Paris, France.
  sanity: min(relevant)=+0.794556 > max(irrelevant)=+0.648514 -> PASS

SANITY: PASS
```

The script exits non-zero if the relevance check fails.

## Results

Measured 2026-11-23 on one Blackhole Galaxy with the image above, 32 model copies behind 16 front ends. Each
request carried one text of 128 tokens; closed loop, 20 s per point, from the image's multi-process load tool
(`serving_study/emb_load_mp.py`, 16 client processes), zero errors at every point.

| Requests in flight | Requests per second | Tokens per second | p50 latency | p90 latency |
|---|---|---|---|---|
| 1 | 101 | n/a | 9.8 ms | 10.1 ms |
| 32 | 3,614 | 462,600 | 8.0 ms | 12.2 ms |
| 256 | 12,712 | 1,627,200 | 18.9 ms | 28.2 ms |
| 1,024 | 14,123 | 1,807,700 | 62.6 ms | 99.9 ms |
| 2,048 | 13,185 | 1,687,700 | 136 ms | 204 ms |

The one-request-in-flight row is from a separate single-process Python client (a new connection per request, texts
of about 128 tokens). Throughput peaks at about 14,000 requests per second around 1,000 requests in flight.

Tenstorrent's validation of this image: cosine similarity of 0.9992 or better (lowest 0.99927) against the Hugging
Face fp32 model, on every chip.

## Notes

- **Prompts:** the server embeds the text it receives. Add EmbeddingGemma's prompts, as the client
  does: `task: search result | query: ` for queries and `title: none | text: ` for documents
  (`config_sentence_transformers.json` lists the other task prompts).
- **Reset with `tt-smi -r all` after every stop**, as for the LLM server.
