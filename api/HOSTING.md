# Self-hosting `himalayagpt-0.5b-it` as an OpenAI-compatible API

This is a step-by-step guide for getting your own deployment up. Two boxes,
two `docker compose up`s, ~30 minutes from zero.

If you just want to try the model locally without any of this, see the
top-level [`README.md`](../README.md) — it covers running `llama-cli`
directly against the GGUFs.

---

## Architecture

```
   ┌───────────────────────────────────────────────────┐
   │                                                   │
   │   client (OpenAI SDK, litellm, curl, …)           │
   │                                                   │
   └───────────────────────┬───────────────────────────┘
                           │  HTTPS, bearer token
                           ▼
   ┌───────────────────────────────────────────────────┐
   │   master  (FastAPI)        ──  $5 VPS             │
   │   Coolify or plain `docker compose up`            │
   │   ▸ /v1/chat/completions  /v1/models  /health     │
   │   ▸ token auth + per-key rate limit               │
   │   ▸ in-memory queue per model slug                │
   └───────────────────────┬───────────────────────────┘
                           ▲
                           │  long-poll, NDJSON-chunked POST
                           │  (worker initiates — no inbound to GPU host)
   ┌───────────────────────┴───────────────────────────┐
   │   worker  (your GPU host)                         │
   │                                                   │
   │   ┌──────────────────────────────────────────┐    │
   │   │ models-init  (alpine + curl, runs once)  │    │
   │   │ downloads the 3 GGUFs to a docker volume │    │
   │   │ if no host path is provided              │    │
   │   └──────────────────────────────────────────┘    │
   │                       │ depends_on: completed     │
   │   ┌──────────────────────────────────────────┐    │
   │   │ llama-server (SYCL build of lukas-h fork)│    │
   │   │ hosts BF16 + Q8_0 + Q4_K_M as 3 slugs    │    │
   │   └──────────────────────────────────────────┘    │
   │                       │ healthcheck              │
   │   ┌──────────────────────────────────────────┐    │
   │   │ agent (Python)                           │    │
   │   │ long-polls master for jobs               │    │
   │   │ proxies to local llama-server, streams   │    │
   │   │ tokens back via chunked HTTP             │    │
   │   └──────────────────────────────────────────┘    │
   └───────────────────────────────────────────────────┘
```

Why split this way?

- **Master is internet-facing** so you can give clients a stable URL with TLS, an API token, and rate limiting. It's stateless and tiny — anywhere with a public IP works.
- **Worker is outbound-only**. Your home GPU never accepts inbound connections — the agent makes outbound long-poll requests to the master. Works behind any NAT / Cloudflare Tunnel / mobile tether.
- **Multi-quant** in one process: `llama-server` runs in router mode hosting BF16 / Q8_0 / Q4_K_M as separate slugs. All three are warm at the same time, switched per request via the `model` field.

---

## Hardware

### Master (the `api/master/` service)

- **Anywhere with a public IPv4 and ≥1 GB RAM**. The master is pure Python + FastAPI; no GPU, no model in memory.
- **Cheapest options that work today (May 2026 prices):**
  - Hetzner Cloud CX22 (€4.59/mo)
  - DigitalOcean Basic Droplet ($4/mo)
  - Vultr Cloud Compute ($3.50/mo)
- A domain (or subdomain) pointing at the VPS. You can also use the IP directly, but TLS is much smoother with a real domain.

### Worker (the `api/worker/` service)

- **A box with a GPU** that has ≥ 4 GB VRAM. Tested on Intel Arc Pro B50; Intel Arc Battlemage, NVIDIA, and AMD GPUs are all supported by the underlying llama.cpp build (you may need to swap the Dockerfile target — see "Other GPUs" below).
- **No public IP required.** The worker only does outbound HTTPS to your master.
- **~5–10 GB free disk** for the docker images (Intel oneAPI base alone is ~3 GB) and ~2 GB for the GGUFs.
- **Reliable internet** is enough — the worker doesn't care about latency to clients, only to your master.

### Both together

If you really want to try this on a single box (your laptop, a single VPS, a Pi cluster, …), the master and the worker both run side-by-side. Set `MASTER_BASE_URL=http://master:8000` in the worker `.env` and put both compose files on the same docker network. You won't get TLS termination but it's a fine smoke-test setup.

---

## 0 · Choose a domain

Pick where the API lives, e.g. `https://gpt.your-domain.com`. Add an `A` record at your DNS provider pointing at the VPS public IP. If you use Cloudflare, both proxied and DNS-only work — but if proxied, increase the proxy idle timeout (default 100 s) on the Enterprise plan or stay under it: master defaults to a 55 s long-poll, well under that ceiling.

---

## 1 · Master (the $5 VPS)

```bash
ssh root@your-vps
git clone --recurse-submodules -b api https://github.com/lukas-h/selfhosted-himalaya-gpt.git
cd selfhosted-himalaya-gpt/api/master

cp .env.example .env
$EDITOR .env
```

Pick values for at least these two:

| Var            | What                                                                                                                |
|----------------|---------------------------------------------------------------------------------------------------------------------|
| `API_TOKENS`   | Comma-separated bearer tokens you'll hand out to clients. Treat as secrets. Example: `momo,client-a-key,client-b-key`. |
| `WORKER_TOKEN` | Single shared secret used by your worker(s) when polling the master. Different from `API_TOKENS`.                   |

Other knobs you can leave at default unless you hit them:

| Var                         | Default                                       | Notes                                                                 |
|-----------------------------|-----------------------------------------------|-----------------------------------------------------------------------|
| `MODEL_SLUGS`               | `himalaya-bf16,himalaya-q8,himalaya-q4`       | Must match section names in `worker/presets/himalaya.ini`.            |
| `MAX_CONCURRENT_PER_WORKER` | `2`                                           | Cap on in-flight jobs per worker. Keep ≤ llama-server's `parallel`.   |
| `MAX_QUEUE_DEPTH`           | `32`                                          | Per-slug queue size before requests get rejected with 503.            |
| `RATE_LIMIT_RPM`            | `60`                                          | Per-key (sha256-hashed bearer) requests/minute.                       |
| `USER_REQUEST_TIMEOUT_S`    | `300`                                         | Master gives up on a queued request after this many seconds.          |
| `WORKER_POLL_TIMEOUT_S`     | `55`                                          | Long-poll deadline; keep under Cloudflare's 100 s default.            |

Bring it up:

```bash
docker compose up -d
docker compose logs -f master   # watch it boot
```

Smoke-test from the VPS:

```bash
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/v1/models -H "Authorization: Bearer <one-of-API_TOKENS>"
```

`/v1/models` will list the three slugs even though no worker has connected yet — they come from `MODEL_SLUGS` in env.

### Adding TLS

Pick whichever fits your setup:

- **Caddy** in front (simplest):
  ```
  gpt.your-domain.com {
      reverse_proxy localhost:8000
  }
  ```
- **Coolify** (what we run): point a Coolify "Docker Compose" application at `selfhosted-himalaya-gpt`'s `api` branch with `Base directory = /api/master`. Set the same env vars in the Coolify UI; Traefik handles TLS automatically.
- **Plain Traefik / Nginx**: standard reverse-proxy setup pointing at `:8000`. Make sure to forward `Host` and `X-Forwarded-For`; the master is launched with `--proxy-headers` so per-key rate limiting then sees the real client IP.

---

## 2 · Worker (the GPU host)

Same repo, different subdir:

```bash
ssh you@your-gpu-host
git clone --recurse-submodules -b api https://github.com/lukas-h/selfhosted-himalaya-gpt.git
cd selfhosted-himalaya-gpt/api/worker

cp .env.example .env
$EDITOR .env
```

Set:

| Var                  | What                                                                                                            |
|----------------------|-----------------------------------------------------------------------------------------------------------------|
| `MASTER_BASE_URL`    | `https://gpt.your-domain.com` — wherever step 1 put the master.                                                 |
| `WORKER_TOKEN`       | Same value as the master's `WORKER_TOKEN`.                                                                      |
| `LLAMA_INTERNAL_KEY` | Random secret. The agent uses it to talk to the local llama-server inside docker (defense in depth, not exposed).|

Optional:

| Var                       | Default                                                                          | Notes                                                              |
|---------------------------|----------------------------------------------------------------------------------|--------------------------------------------------------------------|
| `WORKER_ID`               | container hostname                                                              | Lets you run multiple workers and tell them apart in master logs.  |
| `MAX_CONCURRENT`          | `2`                                                                              | Must be ≤ `parallel` in `presets/himalaya.ini`.                    |
| `MODELS_HOST_PATH`        | unset → fall back to a docker volume                                            | Set to an absolute path that already holds the .gguf files; init container will see them and skip downloading.|
| `MODELS_DOWNLOAD_BASE_URL`| `https://huggingface.co/lukas-h/himalayagpt-0.5b-it-gguf/resolve/main`           | Where the init container downloads from when no host path is given.|
| `RENDER_GID` / `VIDEO_GID`| `992` / `44`                                                                     | Match `getent group render video` on your host. Defaults are stock Ubuntu 24.04. |

Bring it up:

```bash
docker compose up -d
docker compose logs -f         # watch llama build + load the models
```

The first deploy compiles `llama-server` with the SYCL backend inside the Intel oneAPI Docker image. That takes ~10-15 min and pulls a ~2.6 GB base layer. After that the layer is cached and rebuilds are seconds.

You'll see in order:

1. `models-init` runs once, either confirms the bind-mount has the GGUFs or downloads them (first boot only).
2. `llama` builds llama-server, starts, loads all three quants — healthcheck flips to healthy after ~60 s.
3. `agent` connects to `MASTER_BASE_URL` and starts long-polling.

---

## 3 · Verify

From any client with internet:

```bash
curl -fsS https://gpt.your-domain.com/health
# {"status":"ok"}

curl -fsS https://gpt.your-domain.com/readyz
# {"status":"ready"}            ← only after the worker has polled at least once

curl -sS https://gpt.your-domain.com/v1/models \
  -H "Authorization: Bearer <one-of-API_TOKENS>"

curl -sS https://gpt.your-domain.com/v1/chat/completions \
  -H "Authorization: Bearer <one-of-API_TOKENS>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "himalaya-q8",
    "messages": [{"role": "user", "content": "What is the capital of Nepal?"}],
    "max_tokens": 64,
    "temperature": 0.2,
    "top_k": 40,
    "repeat_penalty": 1.08
  }'
```

…or via the OpenAI Python SDK pointed at your domain:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://gpt.your-domain.com/v1",
    api_key="<one-of-API_TOKENS>",
)

resp = client.chat.completions.create(
    model="himalaya-q8",
    messages=[{"role": "user", "content": "Hello in Nepali"}],
    max_tokens=64,
    temperature=0.2,
)
print(resp.choices[0].message.content)
```

Streaming (`stream=True`) works the same.

---

## Other GPUs

The default worker compose builds the SYCL target (`.devops/intel.Dockerfile target: server`) for Intel Arc / Data Center Max / Flex. To use other hardware, switch the `build:` target in `api/worker/docker-compose.yml`:

| GPU vendor   | Dockerfile to use                                | What changes                                                         |
|--------------|--------------------------------------------------|----------------------------------------------------------------------|
| Intel SYCL   | `.devops/intel.Dockerfile` (default)             | nothing                                                              |
| NVIDIA CUDA  | `.devops/cuda.Dockerfile`                        | drop `ONEAPI_DEVICE_SELECTOR`, swap `--gpus all` in for `/dev/dri`   |
| AMD ROCm     | `.devops/rocm.Dockerfile`                        | similar to CUDA, with ROCm runtime                                   |
| CPU only     | `.devops/cpu.Dockerfile`                         | remove the `devices:` and `group_add:` blocks; expect slow inference |

The `lukas-h/llama.cpp` fork's nanochat patches are in the runtime/converter layer, so they apply identically across all backends.

---

## Multiple workers

You can point any number of workers at the same master. Each picks up jobs independently. To distinguish them in logs and concurrency accounting, give each a unique `WORKER_ID`. The master enforces `MAX_CONCURRENT_PER_WORKER` per worker_id, so two workers means twice the in-flight cap automatically.

There's no built-in load balancer — the master just hands a job to whichever worker long-polls first. For a tiny home setup this is enough; if you want fan-out across regions, run several masters behind your own LB.

---

## Common issues

**`/readyz` keeps returning 503 "no recent worker poll"** — the worker hasn't connected yet (still building, hit a build error, or the `WORKER_TOKEN` doesn't match). Check `docker compose logs agent` on the worker side.

**Requests time out at 5 minutes with 504** — same as above; master's accepting requests but no worker is polling for that slug. The 5-minute deadline is `USER_REQUEST_TIMEOUT_S`.

**`Unable to find group render: no matching entries in group file`** — your host's render/video GIDs aren't 992/44. Run `getent group render video` and set `RENDER_GID` / `VIDEO_GID` in `api/worker/.env`.

**Can't write to /models** — `MODELS_HOST_PATH` points at a directory the docker daemon (usually root) can't write to. Either chown the dir to be group-writable by docker's user, or unset `MODELS_HOST_PATH` and let the docker-managed volume handle it.

**SYCL build fails on `oneapi`/`igc` packages** — the pinned base image (`intel/deep-learning-essentials:2025.3.3-0-devel-ubuntu24.04`) is a verified Intel release. If you're on a non-x86 host or an older kernel, see [`llama.cpp/docs/backend/SYCL.md`](../llama.cpp/docs/backend/SYCL.md) and consider switching to the CUDA / CPU target.

**LFS quota errors during clone** — the `api` branch has the GGUFs gitignored to avoid this; if you're somehow on a branch that has them tracked, either `--filter=blob:none` your clone or set `GIT_LFS_SKIP_SMUDGE=1`.

**Cert errors / `502 Bad Gateway`** — TLS is on your reverse proxy (Caddy / Traefik / etc.), not on the master container. Master listens plaintext on `:8000` for the proxy to hit. Make sure the proxy forwards `X-Forwarded-For` or rate limiting becomes per-IP-of-the-proxy (everyone looks the same).

---

## What this is not

- **Production-grade**. The master keeps the queue in-memory; if it restarts, in-flight requests get a 5xx. Acceptable for tiny home use; swap to Redis/SQLite once the load justifies it.
- **A hosted product**. There's no billing, usage tracking, per-user quotas, or admin UI.
- **Tool/function calling capable**. The 0.5B model emits tool tokens (`<|python_*|>`, `<|output_*|>`) but nothing reliable enough to depend on.
- **Multimodal**. Text in, text out.

For everything that *is* solid, see the [`README.md`](../README.md) and [`NANOCHAT_GGUF_HANDOVER.md`](../NANOCHAT_GGUF_HANDOVER.md) at the repo root.
