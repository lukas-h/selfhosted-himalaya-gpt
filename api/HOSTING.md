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
   │   │ downloads the 3 GGUFs to /home/.../...   │    │
   │   │ if the bind-mount dir is empty           │    │
   │   └──────────────────────────────────────────┘    │
   │             │ depends_on: completed               │
   │   ┌──────────────┐ ┌────────────┐ ┌────────────┐  │
   │   │ llama-bf16   │ │ llama-q8   │ │ llama-q4   │  │
   │   │ Vulkan +     │ │ Vulkan +   │ │ Vulkan +   │  │
   │   │ -ngl 999     │ │ -ngl 999   │ │ -ngl 999   │  │
   │   └──────────────┘ └────────────┘ └────────────┘  │
   │             │ healthcheck (each)                  │
   │   ┌──────────────────────────────────────────┐    │
   │   │ agent (Python)                           │    │
   │   │ one puller per slug, long-polls master   │    │
   │   │ routes to llama-{quant}:8080 and streams │    │
   │   │ tokens back via chunked HTTP             │    │
   │   └──────────────────────────────────────────┘    │
   └───────────────────────────────────────────────────┘
```

Why split this way?

- **Master is internet-facing** so you can give clients a stable URL with TLS, an API token, and rate limiting. It's stateless and tiny — anywhere with a public IP works.
- **Worker is outbound-only**. Your home GPU never accepts inbound connections — the agent makes outbound long-poll requests to the master. Works behind any NAT / Cloudflare Tunnel / mobile tether.
- **One container per quant**, not the fork's `--models-preset` router. With a GPU-backed binary the router mode silently zombies child processes ≥2 because each child dlopens the GPU backend and contends over `/dev/dri`. Per-quant containers each get their own process tree, no contention.
- **All-Vulkan backend**: BF16, Q8, and Q4 all use `vulkan.Dockerfile`. SYCL is faster for Q8/Q4 decode in microbenchmarks, but it has repeatedly hit graph/backend stability problems on this model; Vulkan is the simpler stable default.

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

- **A box with a GPU** that has ≥ 4 GB VRAM. Tested on Intel Arc Pro B50 (16 GB); Intel Battlemage, NVIDIA, and AMD GPUs are all supported by the underlying llama.cpp build (you may need to swap the Dockerfile target — see "Other GPUs" below).
- **No public IP required.** The worker only does outbound HTTPS to your master.
- **~5 GB free disk** for the Vulkan docker image and ~2 GB for the GGUFs. More is useful for Docker build cache.
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
| `MODEL_SLUGS`               | `himalaya-bf16,himalaya-q8,himalaya-q4`       | One per llama-{quant} container — slug `himalaya-q8` → `llama-q8:8080`.|
| `MAX_CONCURRENT_PER_WORKER` | `3`                                           | Cap on in-flight jobs per worker. Match worker-side `MAX_CONCURRENT`. |
| `MAX_QUEUE_DEPTH`           | `32`                                          | Per-slug queue size before requests get rejected with 503.            |
| `RATE_LIMIT_RPM`            | `1200`                                        | Sustained per-key (sha256-hashed bearer) requests/minute.             |
| `RATE_LIMIT_BURST_RPS`      | `30`                                          | Per-second burst cap on top of RPM. Set to 0 to disable.              |
| `USER_REQUEST_TIMEOUT_S`    | `240`                                         | Global deadline for every model slug; covers queue wait plus generation time. |
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
| `MAX_CONCURRENT`          | `3`                                                                              | Active-job cap shared across all per-quant pullers in this worker. |
| `LLAMA_URL_TEMPLATE`      | `http://llama-{quant}:8080`                                                      | Slug→URL pattern. `{quant}` = slug with `himalaya-` stripped. Override only if you renamed the per-quant services.|
| `MODELS_DOWNLOAD_BASE_URL`| `https://huggingface.co/lukas-h/himalayagpt-0.5b-it-gguf/resolve/main`           | Where `models-init` downloads from on first boot.                  |
| `RENDER_GID` / `VIDEO_GID`| `992` / `44`                                                                     | Match `getent group render video` on your host. Defaults are stock Ubuntu 24.04. |

The GGUFs live at the hardcoded host path `/home/lukashimsel/himalaya-models` (Coolify's compose parser doesn't support `${VAR}` interpolation in `volumes:` lines). Edit the bind-mount path in `worker/docker-compose.yml` to deploy elsewhere.

Bring it up:

```bash
docker compose up -d
docker compose logs -f         # watch llama build + load the models
```

The first deploy compiles `llama-server` once with the Vulkan backend (~5-10 min). After that the layers are cached and rebuilds are seconds.

You'll see in order:

1. `models-init` runs once, either confirms the bind-mount has the GGUFs or downloads them (first boot only).
2. `llama-bf16`, `llama-q8`, and `llama-q4` build from the Vulkan image, start, and each loads its model — healthchecks flip to healthy after cold start.
3. `agent` waits for all three llama services to be healthy, then connects to `MASTER_BASE_URL` and starts long-polling — one puller per slug. Idle long-polls do not consume active-job capacity.

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

The default worker compose builds one image: `vulkan.Dockerfile` for BF16, Q8, and Q4. To use other hardware, swap the `build.dockerfile` for each `llama-*` service in `api/worker/docker-compose.yml`:

| GPU vendor   | Dockerfile to use            | What changes                                                                                             |
|--------------|------------------------------|----------------------------------------------------------------------------------------------------------|
| Intel Vulkan | `.devops/vulkan.Dockerfile`  | default for all three services. Mesa ANV, no special flags. Works on AMD/NVIDIA via Mesa too.            |
| Intel SYCL   | `.devops/intel.Dockerfile`   | faster q8/q4 decode in microbenchmarks, but unstable on this nanochat graph; requires SYCL-specific workarounds.|
| NVIDIA CUDA  | `.devops/cuda.Dockerfile`    | swap `--gpus all` in for `/dev/dri`. Untested with this fork.                                           |
| AMD ROCm    | `.devops/rocm.Dockerfile`    | similar to CUDA, with ROCm runtime. Untested with this fork.                                             |
| CPU only    | `.devops/cpu.Dockerfile`     | remove the `devices:` and `group_add:` blocks; expect ~80 t/s decode on bf16.                            |

The `lukas-h/llama.cpp` fork's nanochat patches are in the runtime/converter layer, so they apply identically across all backends. The default deploy avoids the SYCL-specific gotchas by using Vulkan for every quant.

---

## Multiple workers

You can point any number of workers at the same master. Each picks up jobs independently. To distinguish them in logs and concurrency accounting, give each a unique `WORKER_ID`. The master enforces `MAX_CONCURRENT_PER_WORKER` per worker_id, so two workers means twice the in-flight cap automatically.

There's no built-in load balancer — the master just hands a job to whichever worker long-polls first. For a tiny home setup this is enough; if you want fan-out across regions, run several masters behind your own LB.

---

## Common issues

**`/readyz` keeps returning 503 "no recent worker poll"** — the worker hasn't connected yet (still building, hit a build error, or the `WORKER_TOKEN` doesn't match). Check `docker compose logs agent` on the worker side.

**Requests time out with 504** — master's accepting requests but no worker is polling for that slug, every worker is saturated, or generation exceeded `USER_REQUEST_TIMEOUT_S`. The default deadline is 240 s and applies to every model slug.

**`HTTP 503 "queue full, try again"`** — the per-slug `MAX_QUEUE_DEPTH=32` is saturated. Either real load is too high (raise the cap) or you fired a burst whose clients abandoned but jobs are still draining.

**`HTTP 429 "rate limit exceeded: N per 1 second/minute"`** — per-token cap; defaults are 30/sec + 1200/min. Bump `RATE_LIMIT_RPM` / `RATE_LIMIT_BURST_RPS` in master env.

**`Unable to find group render: no matching entries in group file`** — your host's render/video GIDs aren't 992/44. Run `getent group render video` and set `RENDER_GID` / `VIDEO_GID` in `api/worker/.env`.

**SYCL build fails on `oneapi`/`igc` packages** — the pinned base image (`intel/deep-learning-essentials:2025.3.3-0-devel-ubuntu24.04`) is a verified Intel release. If you're on a non-x86 host or an older kernel, see [`llama.cpp/docs/backend/SYCL.md`](../llama.cpp/docs/backend/SYCL.md) and consider switching all 3 services to the Vulkan or CPU target.

**LFS quota errors during clone** — root GGUFs are tracked through Git LFS for local runtime use, but `.lfsconfig` excludes `*.gguf` from default fetches so Coolify/API deploys can clone without downloading model blobs. If you actually need the local GGUF files, run `git -c lfs.fetchexclude= lfs pull --include="*.gguf" --exclude=""`.

**Cert errors / `502 Bad Gateway`** — TLS is on your reverse proxy (Caddy / Traefik / etc.), not on the master container. Master listens plaintext on `:8000` for the proxy to hit. Make sure the proxy forwards `X-Forwarded-For` or rate limiting becomes per-IP-of-the-proxy (everyone looks the same).

---

## What this is not

- **Production-grade**. The master keeps the queue in-memory; if it restarts, in-flight requests get a 5xx. Acceptable for tiny home use; swap to Redis/SQLite once the load justifies it.
- **A hosted product**. There's no billing, usage tracking, per-user quotas, or admin UI.
- **A *reliable* function-calling model**. Function/tool calling *is* wired up: the
  master renders OpenAI `tools` into a Hermes prompt (with few-shot priming) and
  parses `<tool_call>` JSON back into OpenAI `tool_calls`
  (`api/master/app/tool_calling.py`). It no longer loops on tool prompts and emits
  well-formed calls — but it's a 0.5B model, so tool *selection* and argument
  accuracy are modest (~40–60% on the bundled eval); treat it as best-effort. See
  `api/tests/test_tool_calling.py` (unit), and `test_tool_calling_eval.py` +
  `tests/bfcl/` (live, env-gated) for what to expect.
- **Multimodal**. Text in, text out.

For everything that *is* solid, see the [`README.md`](../README.md) and [`NANOCHAT_GGUF_HANDOVER.md`](../NANOCHAT_GGUF_HANDOVER.md) at the repo root.
