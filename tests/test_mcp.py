"""Exercise real subprocess stdio and HTTP MCP sessions against an isolated service."""
import asyncio
import os
import socket
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from flux_image.client import Client
from flux_image.config import Settings


@pytest.fixture
def service(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    settings = Settings(home=tmp_path, port=port, token="test-only-secret")
    settings.save()
    env = os.environ.copy()
    env["FLUX_IMAGE_HOME"] = str(tmp_path)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    log = (tmp_path / "server.log").open("wb")
    process = subprocess.Popen([sys.executable, "-m", "flux_image", "serve"], env=env,
                               stdout=log, stderr=log, stdin=subprocess.DEVNULL)
    client = Client(settings, autostart=False)
    try:
        deadline = time.monotonic() + 15
        while not client.reachable():
            if process.poll() is not None or time.monotonic() > deadline:
                raise AssertionError((tmp_path / "server.log").read_text("utf-8"))
            time.sleep(0.1)
        yield settings, env
    finally:
        process.terminate()
        process.wait(timeout=15)
        log.close()


async def exercise(session):
    await session.initialize()
    names = {t.name for t in (await session.list_tools()).tools}
    assert names == {"generate_image", "edit_image", "upload_image", "get_image_job", "cancel_image_job", "image_service_status"}
    result = await session.call_tool("image_service_status", {})
    assert not result.isError
    text = "\n".join(c.text for c in result.content if c.type == "text")
    assert '"upsampler": false' in text
    assert '"cpu_offload": false' in text
    # These tests intentionally have no model weights. A real submission must fail honestly.
    submitted = await session.call_tool("generate_image", {"prompt": "A complete scene caption", "width": 256, "height": 256, "seed": -1})
    assert not submitted.isError
    job_id = submitted.structuredContent["job_id"]
    assert 0 <= submitted.structuredContent["request"]["seed"] < 2**32
    for _ in range(40):
        response = await session.call_tool("get_image_job", {"job_id": job_id})
        if response.structuredContent["status"] == "failed":
            assert "Inference not ready" in response.structuredContent["error"]
            break
        await asyncio.sleep(0.1)
    else:
        raise AssertionError("Missing models did not produce an actionable error")


async def test_stdio_mcp(service):
    settings, env = service
    params = StdioServerParameters(command=sys.executable, args=["-m", "flux_image", "mcp"], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=15)) as session:
            await exercise(session)


async def test_http_mcp(service):
    settings, env = service
    async with httpx.AsyncClient(headers={"Authorization": "Bearer " + settings.token}, trust_env=False) as http:
        async with streamable_http_client(settings.url + "/mcp", http_client=http) as (read, write, _):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=15)) as session:
                await exercise(session)
