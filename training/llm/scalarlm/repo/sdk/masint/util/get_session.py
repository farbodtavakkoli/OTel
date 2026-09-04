from contextlib import asynccontextmanager
import aiohttp
import asyncio

@asynccontextmanager
async def get_session():
    """Context manager for aiohttp sessions with proper cleanup."""
    connector = aiohttp.TCPConnector(force_close=True)
    session = aiohttp.ClientSession(connector=connector)
    try:
        yield session
    finally:
        await session.close()
        await asyncio.sleep(0)  # Extra safety
