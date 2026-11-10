from __future__ import annotations

import json
import os
import secrets
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    home: Path
    port: int = 8765
    gpu_uuid: str = ""
    memory_gib: float = 10.5
    token: str = ""

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def models(self) -> Path:
        return self.home / "models"

    def save(self):
        self.home.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data.pop("home")
        temp = self.home / "config.json.tmp"
        temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temp.replace(self.home / "config.json")


def gpu_inventory() -> list[dict]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return [dict(zip(("uuid", "name", "total_mib", "free_mib"), (x.strip() for x in line.split(",")), strict=True))
                for line in result.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def load_settings() -> Settings:
    home = Path(os.environ.get("FLUX_IMAGE_HOME", ROOT / "data")).resolve()
    home.mkdir(parents=True, exist_ok=True)
    # All processes share this lock, including simultaneous first-use adapters.
    from filelock import FileLock
    with FileLock(str(home / "config.lock")):
        path = home / "config.json"
        data = json.loads(path.read_text("utf-8")) if path.exists() else {}
        settings = Settings(home=home, **data)
        if not settings.token:
            settings.token = secrets.token_urlsafe(32)
        if not settings.gpu_uuid:
            matches = [g for g in gpu_inventory() if "RTX 3060" in g["name"]]
            if len(matches) == 1:
                settings.gpu_uuid = matches[0]["uuid"]
        settings.save()
        return settings


def select_gpu(settings: Settings):
    """Call before importing torch; never silently use another GPU."""
    matches = [g for g in gpu_inventory() if g["uuid"] == settings.gpu_uuid]
    if len(matches) != 1 or "RTX 3060" not in matches[0]["name"]:
        raise RuntimeError("Configured RTX 3060 UUID is unavailable. Run flux-image doctor.")
    os.environ["CUDA_VISIBLE_DEVICES"] = settings.gpu_uuid
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
