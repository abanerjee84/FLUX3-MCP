"""Verify live tool submission, image editing and preview delivery over both MCP transports."""
import asyncio
import json
import os
import sys
import time
from datetime import timedelta

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from flux_image.config import load_settings


async def main():
    settings = load_settings()
    report = json.loads((settings.home / "gpu-validation.json").read_text("utf-8"))
    first = report["cases"][0]["job"]
    prompt = first["request"]["prompt"].replace("matte red", "matte cobalt blue").replace("red body", "cobalt blue body")
    params = StdioServerParameters(command=sys.executable, args=["-m", "flux_image", "mcp"],
                                  env={**os.environ, "FLUX_IMAGE_HOME": str(settings.home)})
    started = time.monotonic()
    async with stdio_client(params) as (read, write), ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30)) as session:
        await session.initialize()
        status = await session.call_tool("image_service_status", {})
        assert status.structuredContent["inference"]["loaded"]
        response = await session.call_tool("edit_image", {"prompt": prompt, "reference_images": [first["job_id"]],
                                                          "width": 512, "height": 512, "seed": 7, "steps": 16})
        assert not response.isError, response
        job_id = response.structuredContent["job_id"]
        print(json.dumps({"stdio_edit_job_id": job_id}), flush=True)
        for _ in range(180):
            response = await session.call_tool("get_image_job", {"job_id": job_id})
            job = response.structuredContent
            if job["status"] in {"succeeded", "failed", "cancelled"}:
                break
            await asyncio.sleep(1)
        assert job["status"] == "succeeded", job
    async with httpx.AsyncClient(headers={"Authorization": "Bearer " + settings.token}, trust_env=False) as http:
        async with streamable_http_client(settings.url + "/mcp", http_client=http) as (read, write, _):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30)) as session:
                await session.initialize()
                result = await session.call_tool("get_image_job", {"job_id": job_id, "include_preview": True})
                assert not result.isError
                assert any(c.type == "image" and c.mimeType == "image/png" for c in result.content)
    result = {"stdio_edit": "passed", "http_image_preview": "passed", "job_id": job_id,
              "elapsed_seconds": round(time.monotonic() - started, 2), "result": job["result"]}
    (settings.home / "mcp-live-validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


asyncio.run(main())
