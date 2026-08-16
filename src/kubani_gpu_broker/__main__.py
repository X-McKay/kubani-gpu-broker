"""Run both listeners: public proxy and internal admin."""

from __future__ import annotations

import asyncio

import uvicorn

from .admin import create_admin_app
from .app import Broker, create_public_app
from .config import load_config


async def _serve() -> None:
    broker = Broker(load_config())
    server_cfg = broker.cfg.server

    public = uvicorn.Server(
        uvicorn.Config(create_public_app(broker), host=server_cfg.host, port=server_cfg.public_port)
    )
    admin = uvicorn.Server(
        uvicorn.Config(
            create_admin_app(broker),
            host=server_cfg.host,
            port=server_cfg.admin_port,
            # Lifespan (broker startup/shutdown) belongs to the public app.
            lifespan="off",
        )
    )
    await asyncio.gather(public.serve(), admin.serve())


def main() -> None:
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
