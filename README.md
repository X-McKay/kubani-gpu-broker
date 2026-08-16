# kubani-gpu-broker

Sleep-aware OpenAI-compatible gateway and exclusive GPU lease broker for
[Kubani](https://github.com/X-McKay/kubani)'s vLLM engines on the DGX Spark.

The design lives in the kubani repo:
`docs/plans/active/2026-08-16-vllm-gpu-sleep-wake-broker.md` — read the
import notes, research findings, and normative amendments at the top of
that document before changing behavior here.

## What it does (target state)

- Proxies `/v1/*` to the vLLM engine (only externally reachable endpoint).
- Sleeps idle engines (vLLM sleep level 1) and transparently wakes them
  on the next request, single-flight.
- Grants exclusive GPU leases to fine-tuning workflows (Temporal), with
  per-lease reclaim mode: `sleep` (~80 GiB freed, seconds to resume) or
  `restart` (full pool, minutes to resume).
- Fails closed on GPU ownership; state reconstructable from Kubernetes.

## Current phase

**Phases 1-3 (engine lifecycle) — implemented.** Byte-faithful streaming
passthrough with in-flight accounting; vLLM sleep driver; transparent
single-flight wake-on-request; feature-flagged idle auto-sleep
(`policies.auto_sleep_enabled`, default off pending GB10 qualification);
authenticated admin listener (`:8081`) with manual sleep/wake and state;
Prometheus metrics. Not yet: Kubernetes Lease, exclusive training lease
API, Temporal activities.

Two listeners (spec section 24.2):

- `:8080` public OpenAI proxy (`/v1/*`, `/healthz`, `/readyz`)
- `:8081` admin (`/metrics`, `/healthz` unauthenticated;
  `/internal/v1/*` requires `Authorization: Bearer $GPU_BROKER_ADMIN_TOKEN`;
  fails closed when the token is unset)

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
```

Run locally against any OpenAI-compatible upstream:

```bash
cat > config.yaml <<'EOF'
engines:
  main:
    base_url: http://localhost:8000
proxy:
  connect_timeout_seconds: 10
EOF
CONFIG_PATH=config.yaml GPU_BROKER_ADMIN_TOKEN=dev uv run python -m kubani_gpu_broker
```

## Deployment

Deployed by Flux from the kubani repo
(`infrastructure/gitops/apps/vllm/broker-*.yaml`); this repo only builds
and publishes the container image to ghcr.io.
