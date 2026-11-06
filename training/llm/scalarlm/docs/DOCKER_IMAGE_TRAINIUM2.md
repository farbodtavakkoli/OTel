# ScalarLM server image: AWS Trainium2

Status: recipe documentation only. No Trainium image was found in the local Docker inventory.
Registry availability, build success and all runtime behavior remain unverified, awaiting
the Trainium agent. The earlier README's Gemma4 test claim is not acceptance evidence.
Client setup and GPU workflows remain in [the client README](../readme_scalarlm.md).
See the [canonical hardware image table](../readme_scalarlm.md#server-image).

## Recipe

| Component | Source contract |
|---|---|
| Serving base | `NEURON_BASE`, supplied Neuron vLLM image; no default or digest pinned |
| Training base | `NEURON_TRAIN_BASE`, must contain `/opt/venv` and `/opt/neuron-beta` |
| Build | `../repo/Dockerfile.trainium`, selected by `TARGET=trainium` |
| Output target | `farbodatdocker/scalarlm:trainium2-v1.0`, not a verified release |
| Server | API, SDK, Slurm and canonical `ml/` copied into `/app/cray` |
| Training | `/opt/venv`; generated `train_job_entrypoint_neuron.sh` sources `neuron_train_env.sh` |
| Defaults | `VLLM_TARGET_DEVICE=neuron`, `SCALARLM_DTYPE=bfloat16`, `NEURON_RT_VIRTUAL_CORE_SIZE=2` |
| Startup | Recipe clears `ENTRYPOINT`; build wrapper adds `CMD ["/app/cray/scripts/start_one_server.sh"]` |

The serving base must provide compatible Neuron runtime libraries and Ubuntu packages,
including `libarchive13t64` and `libpython3.12t64`. The training environment replaces its
library search path with `/opt/aws/neuron/lib`; verify that path in the serving base.
Neither base's compatibility can be inferred from its name.

## Build candidate

Run from `training/llm/scalarlm`. Substitute immutable base image references supplied by
the Trainium agent. The helper synchronizes `ml/` into `repo/ml/` before building.

```bash
cd docs
TARGET=trainium IMAGE_TAG=trainium2-v1.0 \
  NEURON_BASE='<neuron-vllm-image@sha256:digest>' \
  NEURON_TRAIN_BASE='<neuron-pytorch-image@sha256:digest>' \
  ./build_image.sh

docker image inspect farbodatdocker/scalarlm:trainium2-v1.0 \
  --format '{{json .Config.Labels}} {{json .Config.Cmd}} {{json .Config.Entrypoint}}'
```

Release labels must identify the real committed code hash and
`Farbod Tavakkoli <farbodtavakoli@gmail.com>` as both maintainer and author.
The current helper can append a dirty suffix to the revision, including when `REVISION`
is supplied. Such output fails the release provenance gate; resolve that before release.
The helper's final generic GPU acceptance message is not Trainium validation.

## Candidate launch

This preserves the earlier eight device example, not a proven allocation. Replace model,
cache and plugin arguments only after confirming them on the target host.

```bash
docker run -d --name scalarlm \
  --device=/dev/neuron0 --device=/dev/neuron1 --device=/dev/neuron2 --device=/dev/neuron3 \
  --device=/dev/neuron4 --device=/dev/neuron5 --device=/dev/neuron6 --device=/dev/neuron7 \
  --shm-size=64g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 \
  -e SCALARLM_MODEL=/root/models/<serving-model> \
  -e SCALARLM_TENSOR_PARALLEL_SIZE=16 -e SCALARLM_MAX_GPUS_PER_NODE=16 \
  -e SCALARLM_VLLM_ARGS='<vLLM Neuron flags matching the compiled graphs>' \
  -v /path/to/models:/root/models:ro -v /path/to/hf-cache:/root/.cache/huggingface \
  farbodatdocker/scalarlm:trainium2-v1.0

docker logs scalarlm
curl http://localhost:8000/v1/health
```

Device discovery counts four logical cores per visible device. The training environment
selects the last `SLURM_NTASKS_PER_NODE` cores, based on original device numbers.
Preserve host device names. This does not establish that serving uses the first devices
or that concurrent training jobs are isolated; verify both before allowing overlap.

Neuron serving removes GPU specific options, including adapter enabling flags. Use
`--optimization-level=2` with `=` if required by the selected plugin. Compilation duration,
cache paths and cache reuse must be measured, not assumed.

## Candidate client workflow

Run from `training/llm/scalarlm` after client setup. This preserves the earlier Gemma4
example; it does not certify model support or memory fit. Use the small step count for
initial validation before increasing it.

```bash
export SCALARLM_API_URL=http://<trainium-host>:8000
python train.py --training_mode language_model --model_type gemma4 \
  --data_path data/llm_training_sample.json --adapter_type lora --r 32 --lora_alpha 32 \
  --target_modules all-linear --distribution_strategy pytorch_fsdp --reshard_after_forward \
  --llm_name google/gemma-4-31B-it --gpus 16 --nodes 1 --max_steps 2
python inference.py --test_data_path data/llm_training_sample.json \
  --model_name /root/models/<serving-model> --model_type gemma4 --inference_mode language_model
```

`--llm_name` selects the training checkpoint when serving uses a different model.
The inference example addresses the served base, not the training job's adapter.
The earlier `hsdp_shard_size: 16`, sequence 1024 and batch 1 settings are candidates,
not verified requirements. Confirm decoder targets, FSDP behavior and memory use on hardware.

The Neuron branch does not enable live adapter serving. Do not claim job hash inference
or GRPO support. The proposed offline merge workflow remains to be validated:

```bash
cd ml
python -m adapters.merge_lora_and_push --job-dir <job> --repo-id <id> \
  --dry-run --output-dir <dir>
```

After validating export, relaunch a container with `SCALARLM_MODEL` pointing at the merged
model. Confirm any required weight conversion separately. Ask before stopping a container
that this workflow did not start.

## Acceptance gates

| Gate | Evidence required from Trainium agent |
|---|---|
| Build and provenance | Exact base digests, successful build log, output digest and committed revision labels |
| Environment | Serving plugin imports, training Python imports and Neuron driver compatibility |
| Allocation | Actual core visibility, Slurm rank mapping and serving versus training isolation |
| API and serving | Startup logs, health response and a completed generation |
| Training | Small LoRA job with finite loss, intended ranks and a readable checkpoint |
| Resume and export | Checkpoint continuation, merged weights and generation from the merged model |
| Model and compilation | Model revision, dtype, sequence and batch settings, peak memory and cache reuse |

All gates are pending. No Trainium build, container launch, compilation or training was
performed while writing this guide. Do not publish a verified support claim until the
agent supplies these results.
