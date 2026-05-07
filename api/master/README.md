# himalaya-master

Internet-facing OpenAI-compatible API. Authenticates clients, rate-limits, queues jobs, and waits for a worker to pull and process them.

## Endpoints

User-facing (require `Authorization: Bearer <one of API_TOKENS>`):

| Method | Path                     | Notes                                                |
|--------|--------------------------|------------------------------------------------------|
| GET    | `/v1/models`             | Lists the configured slugs.                          |
| POST   | `/v1/chat/completions`   | OpenAI chat completions, both `stream:true` and `false`. |

Worker-facing (require `Authorization: Bearer <WORKER_TOKEN>`):

| Method | Path                                | Notes |
|--------|-------------------------------------|-------|
| GET    | `/internal/jobs/poll`               | Long-poll for the next job. Returns 204 on timeout. |
| POST   | `/internal/jobs/{job_id}/stream`    | NDJSON-framed chunked-body upload of generation events. |

Public (no auth):

| Method | Path        | Notes                                                                  |
|--------|-------------|------------------------------------------------------------------------|
| GET    | `/health`   | Always 200.                                                           |
| GET    | `/readyz`   | 200 if any worker has polled in the last `2 * WORKER_POLL_TIMEOUT_S`.  |

## Configuration

All configuration is via environment variables. See `.env.example` for the full list.

The most important ones:

- `API_TOKENS` — comma-separated bearer tokens accepted on `/v1/*`. Treat as secrets.
- `WORKER_TOKEN` — single shared secret used by the worker on `/internal/*`.
- `MODEL_SLUGS` — must match the per-quant llama services in `api/worker/docker-compose.yml` (`llama-bf16`, `llama-q8`, `llama-q4`).
- `MAX_CONCURRENT_PER_WORKER` — must be `≤` the worker's llama-server `parallel` setting.
- `RATE_LIMIT_RPM` (default `1200`) and `RATE_LIMIT_BURST_RPS` (default `30`, `0` to disable) — both enforced simultaneously per bearer token (sha256-hashed, not per-IP). The tighter window wins per request.
- `USER_REQUEST_TIMEOUT_S` (default `30`) — keep close to typical client timeouts; longer values let abandoned curls saturate the queue with phantom jobs.

## Local run (no Docker)

```bash
cd api/master
uv sync
cp .env.example .env   # then edit it
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Docker

```bash
cp .env.example .env   # set API_TOKENS and WORKER_TOKEN at minimum
docker compose up --build
```

## Deploying to Coolify

Point Coolify at this directory's `docker-compose.yml`. Set `API_TOKENS` and `WORKER_TOKEN` as Coolify secrets. Coolify's Traefik reverse proxy already forwards `X-Forwarded-For`; uvicorn is launched with `--proxy-headers --forwarded-allow-ips '*'` so per-key rate limiting still works.

If Cloudflare is in front of Coolify, two things to know:

- Long-poll lives at 55s by default (`WORKER_POLL_TIMEOUT_S`), well under the 100s Cloudflare idle ceiling.
- The streaming response sends a `: keepalive\n\n` SSE comment every 15s if the model takes a while to start producing tokens.

## Caveats

- **In-memory queue**: master restarts drop in-flight jobs. Users get 5xx and must retry. Acceptable for a tiny home setup; swap to Redis/SQLite if you ever need durability.
- **Single worker assumed**: the `MAX_CONCURRENT_PER_WORKER` semaphore is per-worker-id but the master doesn't load-balance across workers. Multiple workers will work but won't be smart about it.
