"""Collector main entry point."""

import asyncio
from app.collector.runner import main

if __name__ == "__main__":
    asyncio.run(main())
