# himalaya-worker

Runs on the GPU host. Two services in one compose:

- **`llama`**: `llama-server` from the fork, built with SYCL for Intel Arc, hosting all three quants in router mode with a preset INI.
- **`agent`**: small Python service that long-polls the master, forwards each job to local llama-server, streams tokens back via chunked-body POST.

Outbound only — the worker doesn't accept inbound connections from the internet. Cloudflare Tunnels or plain NAT are both fine; the agent only needs to reach the master via outbound HTTPS.

## Hardware assumed

- Intel Arc Pro B50 (or any SYCL-supported Intel GPU with ≥ 4 GB VRAM).
- `/dev/dri/card*` and `/dev/dri/render*` available to the user the docker daemon runs as.

## First-time setup

1. **Pull the GGUFs**. From the repo root: `git lfs pull`. The bind mount at `worker/models/` is symlinked to the three .gguf files at the repo root.
2. **Configure**: `cp worker/.env.example worker/.env` and fill in:
   - `MASTER_BASE_URL` — public URL of your master deployment.
   - `WORKER_TOKEN` — must match the master's `WORKER_TOKEN`.
   - `LLAMA_INTERNAL_KEY` — random secret; only the agent uses it to talk to the local llama-server.

## Build

The first build compiles llama.cpp with SYCL. Takes ~10–15 minutes depending on the host. Subsequent builds use the docker layer cache.

```bash
cd worker
docker compose build
```

If the build fails on `oneapi`/`SYCL` packages, see `llama.cpp/docs/backend/SYCL.md`. The Dockerfile pins to `intel/deep-learning-essentials:2025.3.3-0-devel-ubuntu24.04` which is a verified release.

## Run

```bash
docker compose up
```

`llama` is healthchecked; `agent` only starts after `llama` reports ready (which can take ~60s while it loads all three GGUFs). Once both are up, the agent begins long-polling the master and is ready to serve.

## Verify the GPU

```bash
docker compose exec llama bash -c "ls /dev/dri && sycl-ls"
```

Expect at least one `[level_zero:gpu]` entry for the Arc Pro B50.

## Probe llama-server directly (skip the agent)

Useful while debugging quantization or chat-template issues:

```bash
# from the host
curl -s -H "Authorization: Bearer $LLAMA_INTERNAL_KEY" http://localhost:8080/v1/models   # would need `ports:` to expose 8080; default is `expose:` only
```

If you need to do this regularly, change `expose: ["8080"]` to `ports: ["8080:8080"]` in `docker-compose.yml`. Don't ship that to production — it bypasses the API token and rate limit.

## Caveats

- **KV cache must stay F32**. Don't set `cache-type-k = f16` or `cache-type-v = f16` in `presets/himalaya.ini`. The squared-ReLU MLP overflows F16 — see `../NANOCHAT_GGUF_HANDOVER.md`.
- **`MAX_CONCURRENT` must be `≤` the `parallel` value in `presets/himalaya.ini`** (both default to 2). If higher, llama-server queues internally and you lose visibility.
- **Single worker per master**. Running the same compose on a second box would work but the master doesn't load-balance across workers.
