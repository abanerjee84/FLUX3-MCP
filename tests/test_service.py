import io
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from flux_image.config import Settings
from flux_image.jobs import JobStore
from flux_image.schema import ImageRequest
from flux_image.service import create_app


class TestBackend:
    __test__ = False

    def __init__(self, home):
        self.home = home
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def readiness(self):
        return {"test_backend": True}

    def generate(self, req, job_id, progress, cancelled):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            for n in range(8):
                progress("testing", n * 10)
                time.sleep(0.005)
            path = self.home / "outputs" / f"{job_id}.png"
            Image.new("RGB", (req.width, req.height), (30, 60, 90)).save(path)
            return {"image_path": str(path)}
        finally:
            with self.lock:
                self.active -= 1


def wait(store, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = store.get(job_id)
        if job["status"] in {"succeeded", "cancelled", "failed"}:
            return job
        time.sleep(0.005)
    raise AssertionError("Job did not finish")


def test_queue_serializes_and_idempotent_retry(tmp_path):
    backend = TestBackend(tmp_path)
    store = JobStore(tmp_path, backend)
    req = ImageRequest(prompt="Scene", client_request_id="retry-key")
    first = store.submit(req)
    assert store.submit(req)["job_id"] == first["job_id"]
    with pytest.raises(ValueError, match="different request"):
        store.submit(ImageRequest(prompt="Different", client_request_id="retry-key"))
    jobs = [first] + [store.submit(ImageRequest(prompt=str(i))) for i in range(4)]
    store.start()
    try:
        for job in jobs:
            assert wait(store, job["job_id"])["status"] == "succeeded"
        assert backend.peak == 1
    finally:
        store.close()


def test_cancellation_and_restart_recovery(tmp_path):
    store = JobStore(tmp_path, TestBackend(tmp_path))
    job = store.submit(ImageRequest(prompt="cancel"))
    assert store.cancel(job["job_id"])["status"] == "cancelled"
    abandoned = store.submit(ImageRequest(prompt="abandoned"))
    store.update(abandoned["job_id"], status="running")
    recovered = JobStore(tmp_path, TestBackend(tmp_path))
    assert recovered.get(abandoned["job_id"])["status"] == "failed"
    assert "restarted" in recovered.get(abandoned["job_id"])["error"]


def test_running_cancel_stops_backend(tmp_path):
    backend = TestBackend(tmp_path)
    store = JobStore(tmp_path, backend)
    job = store.submit(ImageRequest(prompt="cancel"))
    store.start()
    try:
        deadline = time.monotonic() + 2
        while store.get(job["job_id"])["status"] == "queued" and time.monotonic() < deadline:
            time.sleep(0.001)
        store.cancel(job["job_id"])
        assert wait(store, job["job_id"])["status"] == "cancelled"
        assert backend.active == 0
    finally:
        store.close()


def test_http_auth_inputs_artifacts_and_limits(tmp_path):
    settings = Settings(home=tmp_path, token="test-secret")
    app = create_app(settings, TestBackend(tmp_path))
    with TestClient(app) as http:
        assert http.get("/health").status_code == 401
        http.headers["Authorization"] = "Bearer test-secret"
        assert http.get("/health").status_code == 200
        assert http.get("/health", headers={"Origin": "https://untrusted.example"}).status_code == 403
        assert http.get("/health", headers={"Host": "untrusted.example"}).status_code == 400
        assert http.post("/v1/jobs", json={"prompt": "test", "width": 2048, "height": 2048}).status_code == 422
        assert http.post("/v1/jobs", json={"prompt": "test", "reference_images": ["../../config"]}).status_code == 422
        assert http.post("/v1/images", content=b"not an image").status_code == 400
        image = io.BytesIO()
        Image.new("RGB", (64, 64)).save(image, format="PNG")
        uploaded = http.post("/v1/images", content=image.getvalue())
        assert uploaded.status_code == 201
        assert http.post("/v1/images", content=image.getvalue()).json()["image_id"] == uploaded.json()["image_id"]
        req = {"prompt": "target scene", "width": 256, "height": 256,
               "reference_images": [uploaded.json()["image_id"]], "client_request_id": "edit-1"}
        response = http.post("/v1/jobs", json=req)
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        assert http.post("/v1/jobs", json=req).json()["job_id"] == job_id
        assert wait(app.state.store, job_id)["status"] == "succeeded"
        assert http.get(f"/v1/jobs/{job_id}/image").headers["content-type"] == "image/png"
        assert http.get("/v1/jobs/missing").status_code == 404


def test_registration_preserves_existing_config(tmp_path):
    import tomlkit

    from flux_image.install import register_codex
    config = tmp_path / "config.toml"
    config.write_text('# Keep comment\nmodel = "existing"\n[mcp_servers.other]\ncommand = "other"\n', encoding="utf-8")
    settings = Settings(home=tmp_path / "data", token="secret")
    register_codex(settings, config)
    register_codex(settings, config)
    contents = config.read_text("utf-8")
    assert "# Keep comment" in contents
    parsed = tomlkit.parse(contents)
    assert parsed["model"] == "existing"
    assert parsed["mcp_servers"]["other"]["command"] == "other"
    assert "local-flux" in parsed["mcp_servers"]
    assert len(list(tmp_path.glob("*.bak"))) == 1


def test_json_registration_preserves_state_and_rejects_collision(tmp_path):
    from flux_image.install import register_json
    config = tmp_path / "claude.json"
    config.write_text(json.dumps({"projects": {"existing": {}}, "mcpServers": {"other": {"command": "other"}}}))
    settings = Settings(home=tmp_path / "data", token="secret")
    register_json(settings, "claude-code", config)
    parsed = json.loads(config.read_text())
    assert parsed["projects"] == {"existing": {}}
    assert parsed["mcpServers"]["other"] == {"command": "other"}
    assert parsed["mcpServers"]["local-flux"]["type"] == "stdio"
    parsed["mcpServers"]["local-flux"]["command"] = "different"
    config.write_text(json.dumps(parsed))
    with pytest.raises(RuntimeError, match="refusing to replace"):
        register_json(settings, "claude-code", config)
