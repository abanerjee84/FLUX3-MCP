"""Opt-in real GPU generation/edit verification through the public local API."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageStat

from flux_image.backend import FluxBackend
from flux_image.client import Client
from flux_image.config import load_settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-models", action="store_true")
    parser.add_argument("--max-wait", type=int, default=3600)
    args = parser.parse_args()
    settings = load_settings()
    deadline = time.monotonic() + args.max_wait
    backend = FluxBackend(settings)
    while not backend.readiness()["ready"]:
        if not args.wait_for_models or time.monotonic() >= deadline:
            raise RuntimeError("Model download or dependencies are incomplete")
        time.sleep(10)
    subprocess.run([sys.executable, "-m", "flux_image", "stop"], check=True)
    client = Client(settings)
    deadline = time.monotonic() + 40
    while client.reachable() and time.monotonic() < deadline:
        time.sleep(0.5)
    client.ensure()
    report = {"started": datetime.now(timezone.utc).isoformat(), "cases": []}
    report_path = settings.home / "gpu-validation.json"

    def run(name, request):
        started = time.monotonic()
        job = client.submit(request)
        print(json.dumps({"case": name, "job_id": job["job_id"]}), flush=True)
        job = client.wait(job["job_id"], timeout=1200)
        case = {"name": name, "elapsed_seconds": round(time.monotonic() - started, 2), "job": job}
        if job["status"] == "succeeded":
            path = Path(job["result"]["image_path"])
            with Image.open(path) as image:
                case["pixel_variance"] = ImageStat.Stat(image.convert("RGB")).var
                assert max(case["pixel_variance"]) > 10, "Output appears blank"
                assert image.size == (request["width"], request["height"])
            memory = job["result"]["memory"]
            assert "3060" in memory["gpu_name"]
            assert memory["residency"]["all_model_tensors_on_cuda"]
            assert memory["peak_reserved_mib"] <= settings.memory_gib * 1024
        report["cases"].append(case)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"case": name, "status": job["status"], "elapsed_seconds": case["elapsed_seconds"],
                          "result": job["result"], "error": job["error"]}), flush=True)
        return job

    prompt = ("Studio product photograph of one matte red ceramic teapot on a pale gray tabletop, "
              "centered in a three-quarter view. A round red body, a curved handle on the left, a short "
              "spout on the right, and a round fitted lid. White seamless background. Soft diffused "
              "light from a large softbox at upper left, a gentle contact shadow, realistic ceramic "
              "texture, crisp edges. No lettering, no other objects.")
    first = run("generate-512", {"prompt": prompt, "width": 512, "height": 512, "steps": 16, "seed": 7})
    if first["status"] != "succeeded":
        raise RuntimeError("Initial GPU generation failed; inspect data/gpu-validation.json")
    reference = client.upload(first["result"]["image_path"])["image_id"]
    edit = prompt.replace("matte red", "matte cobalt blue").replace("red body", "cobalt blue body")
    edited = run("edit-512", {"prompt": edit, "reference_images": [reference], "width": 512, "height": 512,
                              "steps": 16, "seed": 7})
    if edited["status"] != "succeeded":
        raise RuntimeError("GPU editing failed; inspect data/gpu-validation.json")
    for resolution in (768, 1024):
        run(f"generate-{resolution}", {"prompt": prompt, "width": resolution, "height": resolution, "steps": 16, "seed": 7})
    lines = ["# GPU validation", "", f"Started: {report['started']}", "",
             "Official distilled INT8 ConvRot transformer, NF4 text encoder, no upsampler or CPU offload.", "",
             "| Case | Status | Seconds | PyTorch reserved MiB | Driver dedicated MiB | Driver shared MiB |",
             "|---|---|---:|---:|---:|---:|"]
    for case in report["cases"]:
        job = case["job"]
        peak = (job["result"] or {}).get("memory", {}).get("peak_reserved_mib", "")
        driver = (job["result"] or {}).get("memory", {}).get("windows_driver", {})
        lines.append(f"| {case['name']} | {job['status']} | {case['elapsed_seconds']} | {peak} | "
                     f"{driver.get('peak_dedicated_mib', '')} | {driver.get('peak_shared_mib', '')} |")
    lines.extend(["", "Full requests, errors, memory metrics and output paths: `gpu-validation.json`.",
                  "GPU residency was audited for each successful case. Peak metrics are PyTorch allocator measurements;",
                  "they do not include every allocation owned by the Windows driver or other processes."])
    (settings.home / "VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
