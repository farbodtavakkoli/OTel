import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

@asynccontextmanager
async def acquire_file_lock(file_path, timeout=30, poll_interval=0.1):
    file_path = Path(file_path)
    lock_path = file_path.parent / f"{file_path.name}.lock"

    start_time = asyncio.get_event_loop().time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError(f"Could not acquire lock for {file_path} within {timeout} seconds")
            await asyncio.sleep(poll_interval)

    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
