from __future__ import annotations

import asyncio
import logging

import httpx

from .config import settings
from .puller import puller


async def amain() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log = logging.getLogger("worker")
    log.info("worker starting: id=%s slugs=%s concurrency=%d master=%s",
             settings.worker_id, settings.slugs, settings.max_concurrent, settings.master_base_url)

    sem = asyncio.Semaphore(settings.max_concurrent)
    master_headers = {"Authorization": f"Bearer {settings.worker_token}"}
    llama_headers = (
        {"Authorization": f"Bearer {settings.llama_internal_key}"}
        if settings.llama_internal_key
        else {}
    )

    async with httpx.AsyncClient(
        base_url=settings.master_base_url,
        headers=master_headers,
        timeout=httpx.Timeout(connect=15, read=None, write=None, pool=None),
        http2=False,  # keep HTTP/1.1 for chunked upload friendliness through proxies
    ) as master, httpx.AsyncClient(
        base_url=settings.llama_base_url,
        headers=llama_headers,
        timeout=httpx.Timeout(connect=15, read=None, write=None, pool=None),
    ) as llama:
        # one puller per slug = fairness across models, no slug starves another
        await asyncio.gather(*[
            puller(
                slug=slug,
                sem=sem,
                master=master,
                llama=llama,
                worker_id=settings.worker_id,
                poll_timeout_s=settings.poll_timeout_s,
            )
            for slug in settings.slugs
        ])


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
