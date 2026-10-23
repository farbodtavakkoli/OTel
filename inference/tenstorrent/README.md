# `inference/tenstorrent`: Tenstorrent serving with tt-inference-server

Tenstorrent serves models through `tt-inference-server`, which runs vLLM with the
`vllm-tt-plugin` on top of `TT-Metalium`. The endpoint is OpenAI compatible, so the clients
here only need the standard library and `python-dotenv`. The shared setup lives here; the
workload is documented in [`llm/`](llm/).

**Hardware:** Tenstorrent Blackhole Galaxy (32 Blackhole chips, firmware bundle 19.16.0, KMD
2.11.0), verified. Tenstorrent also documents Gemma 4 31B on P300X2 (BH QuietBox 2); we have
not verified it.

## Workloads

| Leaf | Validated model | Port |
|---|---|---|
| [`llm/`](llm/) | `google/gemma-4-31B-it` on Blackhole Galaxy | 8000 |

The `llm/` clients work with any chat model the server runs; pass its name with `--model`.

On Blackhole Galaxy, v0.25.0 serves Gemma 4 31B and 12B, Qwen3 32B and gpt-oss 120B. 

Tenstorrent ships embedding models as tt-metal demos, not servers: `bge-m3` on tt-metal
`main`, and Qwen3-Embedding 0.6B and 4B on branch `arg/embed-4b-pplx-qwen3`. On one
Blackhole chip at 512 tokens we measured 360 embeddings/s for bge-m3, 164 for Qwen3-Embedding-0.6B and 98 for Qwen3-Embedding-4B. Their mean cosine
similarity to the Hugging Face embeddings was 0.938 and 0.949 on a 100 sample MTEB subset.

## What you install

There is no pip wheel for this engine. One prebuilt Docker image holds the whole engine
(TT-Metalium, vllm-tt-plugin and vLLM) and serves the model on port 8000. On the host you
install only the Tenstorrent system software, Docker, and the small client venv below.

Pinned stack, as deployed on our test system:

| Component | Version |
|---|---|
| tt-inference-server | `v0.25.0` (`a822b8fe42903393e7e8dc1bf349ca67ebc4316b`) |
| tt-metal | `978f93b56e36b68318d11ecf7b101085a81d718e` |
| vllm-tt-plugin | `a72303067998a2af060560f9b360061a5fd09d3d` |
| vLLM | `0.26.0` (`system_fingerprint` `vllm-0.26.0-dp4-605eb40c`) |
| Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.25.0-978f93b-a723030` |

The Galaxy config runs 4 data parallel engines of 8 chips each, up to 128 concurrent
requests and a 262,144 token context, with a bfloat8 KV cache and prefix caching on.

## Setup: host

The host needs at least 82 GB of free disk and 77.5 GB of RAM for this model. Install
the system software (firmware, KMD, hugepages and `tt-smi`) with Tenstorrent's installer,
then reboot if it asks you to:

```bash
/bin/bash -c "$(curl -fsSL https://github.com/tenstorrent/tt-installer/releases/latest/download/install.sh)"
```

Check the result. You should see 32 Blackhole chips, firmware bundle 19.16.0, KMD 2.11.0,
and 1G hugepages:

```bash
source ~/.tenstorrent-venv/bin/activate
tt-smi -s --snapshot_no_tty | grep -E '"board_type"|"Driver"' | sort | uniq -c
cat /sys/class/tenstorrent/*/tt_fw_bundle_ver | sort | uniq -c
cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
```

Install Docker and add your user to the `docker` group. Create a Hugging Face read token;
`google/gemma-4-31B-it` is not gated, but a token avoids download rate limits and is
required for gated models.

## Setup: pull and run the image

Pick a model and device pair listed for your hardware in the tt-inference-server
[catalog](https://github.com/tenstorrent/tt-inference-server/blob/v0.25.0/docs/model_support/models_by_hardware.md),
together with the image its model page names. The server refuses any pair that is not in
the catalog. The example is the Gemma 4 31B deployment we validated; its image is public,
about 5.3 GB compressed:

```bash
export MODEL=google/gemma-4-31B-it
export DEVICE=blackhole_galaxy
export IMAGE=ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.25.0-978f93b-a723030
export HF_TOKEN=hf_...
docker pull "$IMAGE"
docker run --rm -d --name tt-inference \
  --env "HF_TOKEN=$HF_TOKEN" \
  --ipc host \
  --publish 8000:8000 \
  --device /dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --volume "volume_id_${MODEL##*/}:/home/container_app_user/cache_root" \
  "$IMAGE" \
  --model "$MODEL" --tt-device "$DEVICE" --no-auth
docker logs -f tt-inference
```

The first start downloads the weights into the named volume and builds the device cache,
so it takes much longer than later starts. The volume keeps both across restarts. To
require an API key, drop `--no-auth` and pass `--env VLLM_API_KEY=<key>`; clients then send
the same `<key>`.

The server is ready when both commands answer:

```bash
curl -s http://localhost:8000/health -o /dev/null -w '%{http_code}\n'
curl -s http://localhost:8000/v1/models
```

Stop it with `docker stop tt-inference`. Because of `--rm`, that also removes the
container; the volume stays.

A clean stop leaves the chips unable to open again: the next server or tt-metal process
fails with `Timed out while waiting for active ethernet core ... Try resetting the board`.
Reset the chips once nothing holds them, before starting the next server:

```bash
source ~/.tenstorrent-venv/bin/activate
tt-smi -r all
```

Alternatively, `run.py` from the same release pulls the image, downloads the weights and
starts the container for you; it logs to `workflow_logs/docker_server/vllm_*.log`. It needs
`python3-venv` and PyYAML on the host:

```bash
sudo apt install -y python3-venv python3-yaml
git clone --branch v0.25.0 --depth 1 https://github.com/tenstorrent/tt-inference-server.git
cd tt-inference-server
python3 run.py --model "$MODEL" --device "$DEVICE" --workflow server \
  --docker-server --override-docker-image "$IMAGE" --no-auth --skip-system-sw-validation
```

The Gemma 4 31B Galaxy deployment uses all 32 chips: start only one server, and keep
everything else off the devices while it runs.

## Reach it over SSH

```bash
ssh -N -L 8000:localhost:8000 user@<galaxy-host>
```

The clients then run on your laptop against `localhost:8000`.

## Client environment

The clients need only `python-dotenv`. Create one venv at the stack root; it serves every
leaf and stays local, outside version control:

```bash
cd inference/tenstorrent
python3 -m venv .env_tenstorrent
source .env_tenstorrent/bin/activate
pip install -r requirements.txt
```

Activate it again in any new shell with `source inference/tenstorrent/.env_tenstorrent/bin/activate`.

Clients load `dev.env` from their own folder (`ln -sf ../../../dev.env dev.env`). With
`--no-auth` no key is needed; if the server was started with `VLLM_API_KEY=<key>`, put
`API_KEY=<key>` in `dev.env`.

## Notes

- **Reasoning is on by default** and counts against `max_tokens`. A small budget can return
  `content: null` with only `reasoning` filled. Raise `max_tokens` or send
  `"chat_template_kwargs": {"enable_thinking": false}`.
- **Give every benchmark request a unique `cache_salt`**, otherwise prefix caching serves
  repeated prompts and you measure the cache.
- **Use `ignore_eos` alone to fix output length.** Adding `min_tokens` halved decode speed
  in our runs.
- **`/v1/completions` does not add `<bos>`.** Raw text prompts degrade; use chat
  completions or send templated token ids.
- **Reset with `tt-smi -r all` after every stop.** Never run `tt-smi -glx_reset`, and
  never reset or open the devices while a server runs: the Galaxy deployment uses all 32
  chips. `tt-smi -r` on a Galaxy needs CPLD firmware 1.16 or newer.
- **Other catalog models use the same image.** Change only `MODEL`. The first start of
  Gemma 4 12B took 6 minutes, Qwen3 32B 30 minutes and gpt-oss 120B 37 minutes; later
  starts reuse the volume.
