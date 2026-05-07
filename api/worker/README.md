# himalaya-worker

Runs on the GPU host. Five services in one compose:

- **`models-init`**: alpine + curl, runs once. Downloads the three GGUFs into a host bind mount if they aren't already there.
- **`llama-bf16`**: `llama-server` built from the fork via `vulkan.Dockerfile`. BF16 with full GPU offload (Mesa ANV → Intel BMG G21).
- **`llama-q8`** and **`llama-q4`**: `llama-server` built via `intel.Dockerfile` (SYCL). Q8_0 and Q4_K_M with full GPU offload via Intel oneAPI / Level Zero. Run with `--batch-size 8 --ubatch-size 8` to dodge a SYCL flash-attn kernel hang on this nanochat compute graph.
- **`agent`**: small Python service. One puller per slug, each long-polling the master and forwarding jobs to its dedicated llama-{quant} container.

Outbound only — the worker doesn't accept inbound connections from the internet. Cloudflare Tunnels or plain NAT are both fine; the agent only needs to reach the master via outbound HTTPS.

## Hardware assumed

- Intel Arc Pro B50 / Battlemage class GPU with ≥ 4 GB VRAM (16 GB on the reference deploy). Total VRAM use across all three containers is ~1.5 GiB model + 360 MiB KV cache.
- `/dev/dri/card*` and `/dev/dri/render*` available to the user the docker daemon runs as.
- ~15 GB free disk for the SYCL image (12.7 GB), the Vulkan image (~715 MB), and the GGUFs (~2 GB).

## First-time setup

1. **Configure**: `cp api/worker/.env.example api/worker/.env` and fill in:
   - `MASTER_BASE_URL` — public URL of your master deployment.
   - `WORKER_TOKEN` — must match the master's `WORKER_TOKEN`.
   - `LLAMA_INTERNAL_KEY` — random secret; only the agent uses it to talk to the local llama-server containers.
2. **(optional) Pre-stage the GGUFs**. Drop them at the host bind path `/home/lukashimsel/himalaya-models/` (edit the path in `docker-compose.yml` if you deploy elsewhere — Coolify's compose parser doesn't expand `${VAR}` in `volumes:` lines). If the dir is empty, `models-init` downloads from `MODELS_DOWNLOAD_BASE_URL` on first boot.

## Build

The first build compiles two images: `intel.Dockerfile` (~20-25 min, pulls the ~3 GB Intel oneAPI base) and `vulkan.Dockerfile` (~5-10 min). Subsequent builds use the docker layer cache.

```bash
cd api/worker
docker compose build
```

If the SYCL build fails on `oneapi`/`igc` packages, see `llama.cpp/docs/backend/SYCL.md`. The pinned base is `intel/deep-learning-essentials:2025.3.3-0-devel-ubuntu24.04`, a verified Intel release.

## Run

```bash
docker compose up
```

`models-init` runs first, then the three llama services start in parallel, then the agent waits for all three healthchecks to flip and begins long-polling. Cold-start is ~60 s for q8/q4 (SYCL kernel JIT) and a few seconds for bf16.

## Verify the GPU

```bash
docker compose exec llama-q8 bash -c "ls /dev/dri && sycl-ls"
docker compose exec llama-bf16 vulkaninfo --summary
```

Expect at least one `[level_zero:gpu]` entry for SYCL and a `Vulkan0` device exposing the Arc Pro B50 for Vulkan.

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
- **SYCL bf16 is broken**. The bf16 model loader splits ~70 % of weights to CPU and segfaults `sched_reserve`. Use `vulkan.Dockerfile` for bf16 (already the default).
- **SYCL q8/q4 needs `--ubatch-size 8`**. The fork's flash-attn SYCL kernel hangs at `batch.n_tokens > ~10` on this compute graph; the strided K^T·Q matmul also crashes when flash-attn is off. Microbatching keeps every physical batch under the threshold without losing GPU-bound throughput.
- **`MAX_CONCURRENT`** is shared across all per-quant pullers — bump it if you want more in-flight headroom, but each llama-server runs `--parallel 1`.
- **Single worker per master**. Running the same compose on a second box would work but the master doesn't load-balance across workers.
