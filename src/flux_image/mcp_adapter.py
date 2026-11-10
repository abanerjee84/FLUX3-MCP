from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, ImageContent, TextContent, ToolAnnotations
from PIL import Image

from .client import Client
from .config import Settings
from .schema import ImageRequest


def structured(value: dict):
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(value))], structuredContent=value)


def create_mcp(settings: Settings, autostart=True):
    client = Client(settings, autostart=autostart)
    mcp = FastMCP("local-flux", host="127.0.0.1", port=settings.port,
        streamable_http_path="/mcp", stateless_http=True, json_response=False,
        instructions=("Local FLUX 3 Klein INT8 generation and editing on the RTX 3060. "
            "Supply a complete final scene caption; there is no prompt upsampler. "
            "Submissions return a job_id immediately. Poll get_image_job until terminal. "
            "Reference paths refer to this computer; upload files first when using HTTP from an isolated client."))

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False))
    async def generate_image(prompt: str, width: int = 768, height: int = 768,
                             steps: int = 16, seed: int | None = None,
                             client_request_id: str | None = None) -> CallToolResult:
        """Queue an image from a complete scene caption. Use seed=-1 or omit for a random seed.

        Returns job_id; no model inference blocks this call.
        """
        req = ImageRequest(prompt=prompt, width=width, height=height, steps=steps, seed=seed,
                           client_request_id=client_request_id)
        return structured(await asyncio.to_thread(client.submit, req.model_dump()))

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False))
    async def upload_image(image_path: str) -> CallToolResult:
        """Copy a local reference image into the service and return image_id. Does not modify the original."""
        return structured(await asyncio.to_thread(client.upload, image_path))

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False))
    async def edit_image(prompt: str, reference_images: list[str], width: int = 768,
                         height: int = 768, steps: int = 16, seed: int | None = None,
                         client_request_id: str | None = None) -> CallToolResult:
        """Queue an edit. References are upload_image IDs, prior successful job IDs, or absolute local image paths.

        Supply a complete target scene caption with the edit incorporated. Up to four references.
        Use seed=-1 or omit for a random seed.
        """
        if not 1 <= len(reference_images) <= 4:
            raise ValueError("Provide one to four reference images")
        ids = []
        for ref in reference_images:
            if len(ref) == 32 and all(c in "0123456789abcdef" for c in ref):
                if (settings.home / "inputs" / f"{ref}.png").is_file():
                    ids.append(ref)
                else:
                    old = await asyncio.to_thread(client.job, ref)
                    if old["status"] != "succeeded":
                        raise ValueError("Reference job has not succeeded")
                    uploaded = await asyncio.to_thread(client.upload, old["result"]["image_path"])
                    ids.append(uploaded["image_id"])
            else:
                path = Path(ref)
                if not path.is_absolute():
                    raise ValueError("Local reference paths must be absolute")
                uploaded = await asyncio.to_thread(client.upload, ref)
                ids.append(uploaded["image_id"])
        req = ImageRequest(prompt=prompt, width=width, height=height, steps=steps,
                           seed=seed, reference_images=ids, client_request_id=client_request_id)
        return structured(await asyncio.to_thread(client.submit, req.model_dump()))

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    async def get_image_job(job_id: str, include_preview: bool = False) -> CallToolResult:
        """Get queued/running/succeeded/failed/cancelled status, errors, output path and memory metrics.

        Set include_preview=true after success to attach a PNG preview for visual inspection.
        """
        job = await asyncio.to_thread(client.job, job_id)
        content = [TextContent(type="text", text=json.dumps(job))]
        if include_preview and job["status"] == "succeeded":
            data = await asyncio.to_thread(client.image, job_id)
            with Image.open(io.BytesIO(data)) as image:
                image.thumbnail((768, 768))
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
            content.append(ImageContent(type="image", data=base64.b64encode(buffer.getvalue()).decode(), mimeType="image/png"))
        return CallToolResult(content=content, structuredContent=job)

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False))
    async def cancel_image_job(job_id: str) -> CallToolResult:
        """Cancel a pending job or request cancellation at the next inference boundary."""
        return structured(await asyncio.to_thread(client.cancel, job_id))

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    async def image_service_status() -> CallToolResult:
        """Report the selected GPU, model availability, residency policy and latest peak memory."""
        return structured(await asyncio.to_thread(client.health))

    return mcp
