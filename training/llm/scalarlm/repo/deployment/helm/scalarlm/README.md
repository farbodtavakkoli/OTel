# ScalarLM Helm chart

One parameterized chart for every ScalarLM deployment. Per-model overrides
live under [`values/`](./values/).

## Quick start

```bash
helm install gemma4 ./deployment/helm/scalarlm \
  -n scalarlm --create-namespace \
  -f ./deployment/helm/scalarlm/values/gemma4-31b.yaml \
  --set cloudflared.tunnelToken=$CF_TOKEN
```

After the rollout, verify health:

```bash
helm test gemma4 -n scalarlm
```

The test hook curls `/v1/health` and fails the install unless
`{"all":"up"}` comes back.

## What you configure

| Knob                            | Values                                 |
|---------------------------------|----------------------------------------|
| `model`                         | HF model ID                            |
| `replicaCounts.inference`       | vLLM replicas (set 0 to disable)       |
| `replicaCounts.training`        | Megatron StatefulSet replicas          |
| `storage.cache.kind`            | `pvc` \| `hostPath` \| `emptyDir`      |
| `storage.jobs.size` / `storageClass` | training job outputs            |
| `vllm.extraArgs`                | passthrough to `SCALARLM_VLLM_ARGS`    |
| `vllm.extraEnv` / `megatron.extraEnv` / `api.extraEnv` | extra env    |
| `megatron.hostIPC`              | required by some NCCL / RDMA setups    |
| `megatron.dshm.kind`            | `disabled` \| `emptyDir` \| `hostPath` |
| `megatron.extraResources`       | e.g. `rdma/hca: 1`                     |
| `megatron.extraCapabilities`    | add to `securityContext.capabilities`  |
| `api.hostPort`                  | expose api + ui on node hostPort       |
| `cloudflared.enabled` + `tunnelToken` | optional tunnel (token in Secret) |
| `*.nodeAffinityHostnames`       | preferred node hostnames               |
| `extraConfig`                   | merged into every cray-config.yaml     |

See [`values.yaml`](./values.yaml) for the annotated defaults.

## Health and deployment checks

Every workload ships with:

- **startupProbe** — generous failure threshold so first-time model pulls
  don't kill the pod. Tune via `*.probes.startup.periodSeconds` and
  `failureThreshold`.
- **readinessProbe** — gates the Service; removed from the endpoints
  during model reloads.
- **livenessProbe** — restarts a hung pod.

| Service  | Probe                             |
|----------|-----------------------------------|
| api      | HTTP `GET /v1/health` on 8000     |
| vllm     | HTTP `GET /health` on 8001        |
| megatron | TCP on slurmd port 6818           |

Plus a [`helm test`](./templates/tests/health.yaml) hook that asserts
`/v1/health` reports `"all":"up"`.

## Secrets

The Cloudflare tunnel token is written to a Kubernetes Secret and read by
`cloudflared` via a `valueFrom.secretKeyRef`. It's **not** passed as a
CLI arg (where it would show up in `ps` and `kubectl describe`) and
**not** baked into the chart.

For GitOps workflows, pre-create a Secret with key `tunnelToken` and set
`cloudflared.existingSecret: <name>`.

## Adding a new model

Copy one of the existing files in `values/`, edit, and `helm install`:

```bash
cp values/qwen3-14b.yaml values/my-model.yaml
# edit model:, image.tag, resource knobs
helm install my-model . -f values/my-model.yaml \
  --set cloudflared.tunnelToken=$CF_TOKEN
```
