from __future__ import annotations

import hashlib
import hmac
import io
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.datastructures import Headers
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .backend import FluxBackend
from .config import Settings
from .jobs import JobStore
from .schema import ImageRequest

MAX_UPLOAD = 32 * 1024 * 1024


class LocalAuthMiddleware:
    """Pure ASGI middleware preserves MCP's streaming response task lifecycle."""
    def __init__(self, app, settings: Settings):
        self.app, self.settings = app, settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = Headers(scope=scope)
        supplied = headers.get("authorization", "")
        if not hmac.compare_digest(supplied.encode(), ("Bearer " + self.settings.token).encode()):
            return await JSONResponse({"detail": "Bearer token required"}, status_code=401)(scope, receive, send)
        origin = headers.get("origin")
        if origin and origin not in {self.settings.url, f"http://localhost:{self.settings.port}"}:
            return await JSONResponse({"detail": "Origin rejected"}, status_code=403)(scope, receive, send)
        await self.app(scope, receive, send)


def create_app(settings: Settings, backend=None) -> FastAPI:
    from .mcp_adapter import create_mcp
    backend = backend or FluxBackend(settings)
    store = JobStore(settings.home, backend)
    mcp = create_mcp(settings, autostart=False)
    mcp_app = mcp.streamable_http_app()

    @asynccontextmanager
    async def lifespan(app):
        store.start()
        try:
            async with mcp.session_manager.run():
                yield
        finally:
            store.close()

    app = FastAPI(title="FLUX Image Local", version="0.1.0", lifespan=lifespan)
    app.state.store = store
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    app.add_middleware(LocalAuthMiddleware, settings=settings)

    @app.get("/health")
    def health():
        return {"service": "flux-image-local", "version": "0.1.0", "home": str(settings.home),
                "inference": backend.readiness()}

    @app.post("/admin/shutdown")
    def shutdown(tasks: BackgroundTasks):
        callback = getattr(app.state, "shutdown", None)
        if callback is None:
            raise HTTPException(409, "This service is externally managed")
        tasks.add_task(callback)
        return {"status": "stopping"}

    @app.post("/v1/jobs", status_code=202)
    def submit(req: ImageRequest):
        for image_id in req.reference_images:
            if not (settings.home / "inputs" / f"{image_id}.png").is_file():
                raise HTTPException(404, f"Reference image {image_id} is not uploaded")
        try:
            return store.submit(req)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        except OverflowError as exc:
            raise HTTPException(429, str(exc)) from exc

    @app.get("/v1/jobs/{job_id}")
    def status(job_id: str):
        try:
            return store.get(job_id)
        except KeyError:
            raise HTTPException(404, "Unknown job") from None

    @app.post("/v1/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        try:
            return store.cancel(job_id)
        except KeyError:
            raise HTTPException(404, "Unknown job") from None

    @app.get("/v1/jobs/{job_id}/image")
    def output(job_id: str):
        job = status(job_id)
        if job["status"] != "succeeded":
            raise HTTPException(409, "Image is not ready")
        # Only use the server-issued UUID, never an input path or arbitrary DB result path.
        if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
            raise HTTPException(404)
        return FileResponse(settings.home / "outputs" / f"{job_id}.png", media_type="image/png")

    @app.post("/v1/images", status_code=201)
    async def upload(request: Request):
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_UPLOAD:
                raise HTTPException(413, "Image upload exceeds 32 MB")
            chunks.append(chunk)
        try:
            with Image.open(io.BytesIO(b"".join(chunks))) as original:
                if original.width * original.height > 16 * 1024 * 1024:
                    raise HTTPException(413, "Input exceeds 16 megapixels")
                image = ImageOps.exif_transpose(original).convert("RGB")
                image.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise HTTPException(400, "Invalid image") from exc
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = buffer.getvalue()
        image_id = hashlib.sha256(encoded).hexdigest()[:32]
        target = settings.home / "inputs"
        target.mkdir(exist_ok=True)
        destination = target / f"{image_id}.png"
        if not destination.exists():
            # Replace atomically; simultaneous identical uploads produce the same asset ID.
            import uuid
            temp = target / f"{image_id}.{uuid.uuid4().hex}.tmp"
            temp.write_bytes(encoded)
            temp.replace(destination)
        return {"image_id": image_id, "width": image.width, "height": image.height}

    app.mount("/", mcp_app)
    return app
