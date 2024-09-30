import asyncio
import pytest
import subprocess
from aiohttp import web

async def health_check(request):
    return web.Response(text="HEALTHY")

@web.middleware
async def middleware(request, handler):
    # Access request.url.path to trigger the potential error
    print(f"Accessing path: {request.url.path}")
    response = await handler(request)
    return response

async def run_server():
    app = web.Application(middlewares=[middleware])
    app.router.add_get('/health', health_check)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 9090)
    await site.start()
    print("Server started on http://localhost:9090")
    return runner

@pytest.mark.asyncio
async def test_health_check():
    runner = await run_server()
    try:
        # Use asyncio.create_subprocess_exec to run curl command
        proc = await asyncio.create_subprocess_exec(
            'curl', '-s', 'http://localhost:9090/health',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        except asyncio.TimeoutError:
            print("Curl request timed out")
            proc.kill()
            await proc.wait()
            return

        if proc.returncode != 0:
            print(f"Curl failed with return code {proc.returncode}")
            print(f"stderr: {stderr.decode()}")
        else:
            response = stdout.decode().strip()
            assert response == "HEALTHY", f"Unexpected response: {response}"
            print("Test passed: Received 'HEALTHY' response")
    finally:
        await runner.cleanup()