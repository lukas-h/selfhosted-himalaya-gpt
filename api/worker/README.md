# himalaya-worker

Runs on the GPU host. Five services in one compose:

- **`models-init`**: alpine + curl, runs once. Downloads the three GGUFs into a host bind mount if they aren't already there.
- **`llama-bf16`**, **`llama-q8`**, and **`llama-q4`**: `llama-server` built from the fork via `vulkan.Dockerfile`, with full GPU offload through Mesa ANV on the reference Intel Arc Pro B50 deploy.
- **`agent`**: small Python service. One puller per slug, each long-polling the master and forwarding jobs to its dedicated llama-{quant} container.

Outbound only — the worker doesn't accept inbound connections from the internet. Cloudflare Tunnels or plain NAT are both fine; the agent only needs to reach the master via outbound HTTPS.

## Hardware assumed

- Intel Arc Pro B50 / Battlemage class GPU with ≥ 4 GB VRAM (16 GB on the reference deploy). Total VRAM use across all three containers is ~1.5 GiB model + 360 MiB KV cache.
- `/dev/dri/card*` and `/dev/dri/render*` available to the user the docker daemon runs as.
- ~5 GB free disk for the Vulkan image, build cache, and the GGUFs (~2 GB).

## First-time setup

1. **Configure**: `cp api/worker/.env.example api/worker/.env` and fill in:
   - `MASTER_BASE_URL` — public URL of your master deployment.
   - `WORKER_TOKEN` — must match the master's `WORKER_TOKEN`.
   - `LLAMA_INTERNAL_KEY` — random secret; only the agent uses it to talk to the local llama-server containers.
2. **(optional) Pre-stage the GGUFs**. Drop them at the host bind path `/home/lukashimsel/himalaya-models/` (edit the path in `docker-compose.yml` if you deploy elsewhere — Coolify's compose parser doesn't expand `${VAR}` in `volumes:` lines). If the dir is empty, `models-init` downloads from `MODELS_DOWNLOAD_BASE_URL` on first boot.

## Build

The first build compiles one Vulkan image (~5-10 min). Subsequent builds use the docker layer cache.

```bash
cd api/worker
docker compose build
```

If Vulkan device discovery fails, check `/dev/dri` passthrough and the host render/video group IDs.

## Run

```bash
docker compose up
```

`models-init` runs first, then the three Vulkan-backed llama services start in parallel, then the agent waits for all three healthchecks to flip and begins long-polling. The pullers do not consume the global active-job semaphore while idle; capacity is held only while a llama request is running.

## Verify the GPU

```bash
docker compose exec llama-q8 bash -c "ls /dev/dri && vulkaninfo --summary"
docker compose exec llama-bf16 vulkaninfo --summary
```

Expect a `Vulkan0` device exposing the Arc Pro B50 or your selected GPU.

## Probe llama-server directly (skip master + agent)

Useful while debugging quantization or chat-template issues:

```bash
# Inside the worker docker network — agent talks to it on this URL too:
docker compose exec agent curl -s \
  -H "Authorization: Bearer $LLAMA_INTERNAL_KEY" \
  http://llama-q8:8080/v1/models
```

To hit it from the host, add `ports: ["127.0.0.1:9002:8080"]` to whichever `llama-*` service you want to probe. Don't ship that to production — it bypasses the API token and rate limit.

## Caveats

- **KV cache must stay F32**. Don't set `cache-type-k = f16` or `cache-type-v = f16`. The squared-ReLU MLP overflows F16 — see `../../NANOCHAT_GGUF_HANDOVER.md`.
- **SYCL is not the default**. It has faster q8/q4 decode in microbenchmarks, but has repeatedly hit backend stability problems on this graph. Keep the default all-Vulkan setup unless you are intentionally benchmarking SYCL.
- **`MAX_CONCURRENT`** defaults to `3`, one active job for each per-quant llama service. Keep it aligned with the master's `MAX_CONCURRENT_PER_WORKER`.
- **Single worker per master**. Running the same compose on a second box would work but the master doesn't load-balance across workers.
