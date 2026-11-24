# `inference/tenstorrent/decider`: strands-decider served on Tenstorrent Blackhole Galaxy

Talks to the Tenstorrent decider server through its System One endpoint, `POST /v1/systemone`.
We validated it with `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` on Blackhole Galaxy,
data parallel over all 32 chips. The decider answers questions about a state (a message, a
conversation or a JSON record) in one forward pass: yes/no questions with a probability,
multiple-choice questions with a probability per option, and ordinal scores.
`decide_tenstorrent.py` sends four built-in telecom requests (a fiber outage, a duplicate charge,
a plan inquiry and a cell-site alarm), checks the answers it expects, and writes a JSON reference
artifact.

**Hardware:** Tenstorrent Blackhole Galaxy, 32 Blackhole chips (firmware bundle 19.12.0, KMD
2.7.0 on our test system), verified with one model copy per chip (DP=32).

## Files

- `decide_tenstorrent.py`: System One client with a built-in answer check.

## Setup: server

| Component | Version |
|---|---|
| Image | `ghcr.io/tenstorrent/tt-inference-server/tt-media-inference-server-strands-decider:0.26.1-0c6f105a` (public) |
| Digest | `sha256:25134cd353778e5aef3186a130802cf988bad1dbdba256a6a77c0a7a213a513c` |
| tt-metal | `0c6f105a3ecc7c21f6f742f12a7a8ee112d6d90a` (branch `arg/strands-decider-dp32`) |
| Decider revision | `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` at `4d27d8d7edcfd6616806c8684e20e2673684e283` |
| Base model revision | `Qwen/Qwen3.5-2B-Base` at `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` |

Install the host software as in [`../README.md`](../README.md). Download the pinned decider
and its base model (both public, about 4.5 GB together) into a work folder, then let the server
(user 1000) write there:

```bash
export WORK=$HOME/decider && mkdir -p $WORK/hf $WORK/tt_cache
HF_HOME=$WORK/hf hf download StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --revision 4d27d8d7edcfd6616806c8684e20e2673684e283
HF_HOME=$WORK/hf hf download Qwen/Qwen3.5-2B-Base --revision b1485b2fa6dfa1287294f269f5fb618e03d52d7c
chmod -R a+rwx $WORK
```

Start the server on all 32 chips. It runs one decider per chip behind 8 FastAPI front ends that share
port 8000; an idle chip takes the next batch of questions, whichever front end received them:

```bash
IMG=ghcr.io/tenstorrent/tt-inference-server/tt-media-inference-server-strands-decider:0.26.1-0c6f105a
SNAP=/work/hf/hub
docker run -d --name decider-dp32 --network host --user 1000:1000 --shm-size 16g --cap-add SYS_NICE \
  --device /dev/tenstorrent -v /dev/hugepages-1G:/dev/hugepages-1G -v $WORK:/work \
  -e HF_HOME=/work/hf -e HF_HUB_OFFLINE=1 -e TT_CACHE_PATH=/work/tt_cache \
  -e STRANDS_DECIDER_MODEL=$SNAP/models--StrandsAgents--strands-decider-2B-qwen3.5-v1-2610/snapshots/4d27d8d7edcfd6616806c8684e20e2673684e283 \
  -e HF_MODEL=$SNAP/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c \
  --entrypoint bash $IMG \
  -c 'source $TT_METAL_HOME/python_env/bin/activate && cd $TT_METAL_HOME && python -m models.demos.strands_agent.serve.server --host 0.0.0.0 --port 8000 --chips all --frontends 8'
```

The first chip starts alone: it converts the weights into `$WORK/tt_cache` and compiles the
kernels, then the other 31 start together. Later starts reuse `$WORK/tt_cache` and were ready in about 260 s in our runs. The server is ready when `/health`
reports all 32 workers:

```bash
curl -s localhost:8000/health   # "status": "ok", "workers": 32, "workers_ready": 32
```

`/health` returns `"status": "starting"` (HTTP 503) until the first chip is ready; requests are
served from then on, by the chips that are ready.

Stop it with `docker rm -f decider-dp32`, then reset the chips with `tt-smi -r all` before the
next server, as in [`../README.md`](../README.md).

## Client

```bash
source ../.env_tenstorrent/bin/activate
ln -sf ../../../dev.env dev.env
python decide_tenstorrent.py --output $OUTPUT_DIR/inference_decider_tenstorrent/reference_decider.json
python decide_tenstorrent.py --request_file my_request.json
```

Raw equivalent:

```bash
curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "My home fiber has been down since yesterday and I work from home. Send a technician today!",
  "questions": {
    "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
    "team": {"type": "choice", "instructions": "Which team should handle this message?",
             "criteria": {"billing": "charges and refunds", "technical_support": "outages and faults", "sales": "plans and pricing"}},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["calm", "frustrated", "furious"]}
  }
}'
```

| Question type | `criteria` | Answer fields |
|---|---|---|
| `noul` (yes/no) | optional `{"false": ..., "true": ...}` | `noul`: probability of yes |
| `choice` | `{option: description, ...}` | `choice`, `probabilities`, `confidence` |
| `score` | `[level 0, level 1, ...]` | `score` (expected level), `legend`, `probabilities`, `confidence` |

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `localhost` | Server host |
| `--port` | `8000` | Server port |
| `--request_file` | `None` | JSON request `{state, questions}` or a list of them (omit for the built-in cases) |
| `--output` | `None` | Path for the JSON reference artifact |
| `--timeout` | `120.0` | HTTP timeout in seconds |

Output on one Blackhole Galaxy, all 32 chips:

```text
endpoint      : http://localhost:8000/v1/systemone
model         : strands-decider-2B-qwen3.5-v1-2610
workers       : 32/32 ready on blackhole 1x1 x 32

outage: 36.8 ms round trip, server 35.3 ms, 366 input tokens
  team               choice technical_support (confidence 0.99)  expect technical_support -> PASS
  urgent             noul 0.874  expect True -> PASS
  frustration        score 1.11 of 0..2

double_charge: 22.1 ms round trip, server 20.41 ms, 235 input tokens
  team               choice billing (confidence 0.97)  expect billing -> PASS
  refund             noul 0.883  expect True -> PASS

plan_inquiry: 21.7 ms round trip, server 20.4 ms, 238 input tokens
  team               choice sales (confidence 0.90)  expect sales -> PASS
  urgent             noul 0.114  expect False -> PASS

network_alarm: 28.7 ms round trip, server 27.12 ms, 353 input tokens
  action             choice dispatch_field_tech (confidence 0.50)  expect dispatch_field_tech -> PASS
  service_impacting  noul 0.757  expect True -> PASS

SANITY: PASS
```

The script exits non-zero if an expected answer is wrong.

## Results

Measured 2026-11-24 on one Blackhole Galaxy with the image above, 32 chips behind 8 front ends. Each request
asked one question about a state of 128 tokens; closed loop, from the image's load tool
(`models/demos/strands_agent/serve/client.py`, 16 client processes on the same host), zero errors at every point.

| Requests in flight | Requests per second | p50 latency | p90 latency |
|---|---|---|---|
| 1 | 57 | 17.5 ms | 18.0 ms |
| 32 | 1,928 | 15.5 ms | 17.0 ms |
| 128 | 3,000 | 21.2 ms | 94.1 ms |
| 256 | 3,143 | 42.6 ms | 168.3 ms |

Throughput peaks at about 3,100 requests per second. At 512 and 1,024 requests in flight the load tool, which shares
the host's CPU with the server, measured 2,829 and 1,928 requests per second, still with zero errors. A rerun on a
fresh start of the published image measured 1,968 requests per second at 32 in flight (p50 15.5 ms) and 2,960 at 256
(p50 56.3 ms).

One request stream (one client process):

| Request | Latency (p50) | Throughput |
|---|---|---|
| 1 question, 128-token state | 17.5 ms | 57 requests/s |
| 5 questions, 128-token state | 43.0 ms | 116 questions/s |
| 1 question, 1,024-token state | 44.8 ms | 22 requests/s |

## Notes

- **Questions:** each question about a state is one row of a device batch. In our runs a request with five
  questions took 43 ms, against 17.5 ms for one question (see Results).
- **Context:** each question sees up to 4096 tokens (state plus question); the server truncates longer states.
- **`/metrics`** reports latency percentiles and batch sizes of the front end that answers the request.
- **Reset with `tt-smi -r all` after every stop**, as for the LLM server.
