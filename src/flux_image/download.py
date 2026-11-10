"""Resumable, bounded-parallel HTTPS ranges with final SHA-256 verification."""
from __future__ import annotations

import hashlib
import json
import math
import os
import ssl
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import truststore
from filelock import FileLock


def verified_download(repo: str, filename: str, revision: str, destination: Path):
    from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url
    url = hf_hub_url(repo, filename, revision=revision)
    meta = get_hf_file_metadata(url)
    size, digest = meta.size, meta.etag.strip('"')
    if not size or len(digest) != 64:
        raise RuntimeError(f"Expected a SHA-256-addressed model file: {filename}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".download.lock", timeout=600):
        verified = destination.with_suffix(destination.suffix + ".verified.json")
        identity = {"size": size, "sha256": digest, "repo": repo, "revision": revision, "file": filename}
        if destination.is_file() and destination.stat().st_size == size:
            if verified.is_file() and json.loads(verified.read_text("utf-8")) == identity:
                return destination
            if file_sha256(destination) == digest:
                verified.write_text(json.dumps(identity), encoding="utf-8")
                return destination
        part = destination.with_suffix(destination.suffix + ".part")
        resume = destination.with_suffix(destination.suffix + ".resume.json")
        chunk_size = 16 * 1024 * 1024
        total = math.ceil(size / chunk_size)
        old = json.loads(resume.read_text("utf-8")) if resume.is_file() else {}
        if old.get("identity") != identity or not part.is_file() or part.stat().st_size != size:
            with part.open("wb") as output:
                if os.name == "nt":
                    # Avoid writing gigabytes of zeros before the first range arrives on NTFS.
                    import ctypes
                    import msvcrt
                    from ctypes import wintypes
                    control = ctypes.windll.kernel32.DeviceIoControl
                    control.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                        wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
                    control.restype = wintypes.BOOL
                    returned = wintypes.DWORD()
                    control(msvcrt.get_osfhandle(output.fileno()), 0x900C4, None, 0, None, 0, ctypes.byref(returned), None)
                output.truncate(size)
            old = {"identity": identity, "complete": []}
        complete = set(old["complete"])
        lock = threading.Lock()
        token = get_token()
        headers = {"Authorization": "Bearer " + token} if token else {}
        workers = min(8, max(1, int(os.environ.get("FLUX_DOWNLOAD_WORKERS", "6"))))
        print(f"Downloading {filename}: {size / 1024**3:.2f} GiB ({len(complete)}/{total} ranges cached)", file=sys.stderr, flush=True)
        with httpx.Client(verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
                          timeout=httpx.Timeout(30, read=120), follow_redirects=True,
                          limits=httpx.Limits(max_connections=workers)) as client:
            def fetch(index):
                start, end = index * chunk_size, min(size, (index + 1) * chunk_size) - 1
                expected_range = f"bytes {start}-{end}/{size}"
                for attempt in range(5):
                    try:
                        response = client.get(url, headers={**headers, "Range": f"bytes={start}-{end}"})
                        if response.status_code != 206 or response.headers.get("content-range") != expected_range:
                            raise RuntimeError(f"Invalid range response ({response.status_code})")
                        content = response.content
                        if len(content) != end - start + 1:
                            raise RuntimeError("Incomplete byte range")
                        with part.open("r+b") as output:
                            output.seek(start)
                            output.write(content)
                            output.flush()
                        with lock:
                            complete.add(index)
                            temp = resume.with_suffix(".tmp")
                            temp.write_text(json.dumps({"identity": identity, "complete": sorted(complete)}), encoding="utf-8")
                            temp.replace(resume)
                            if len(complete) % 16 == 0 or len(complete) == total:
                                print(f"  {filename}: {len(complete)}/{total} ranges", file=sys.stderr, flush=True)
                        return
                    except (httpx.HTTPError, RuntimeError) as exc:
                        if attempt == 4:
                            raise RuntimeError(f"Download failed for {filename} range {index}: {type(exc).__name__}; rerun to resume") from None
                        time.sleep(attempt + 1)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(fetch, [i for i in range(total) if i not in complete]))
        print(f"Verifying SHA-256: {filename}", file=sys.stderr, flush=True)
        if file_sha256(part) != digest:
            # Retain the bad file for inspection, but do not mark its ranges reusable.
            resume.write_text(json.dumps({"identity": identity, "complete": []}), encoding="utf-8")
            raise RuntimeError(f"SHA-256 mismatch for {filename}; the file was not installed")
        part.replace(destination)
        verified.write_text(json.dumps(identity, indent=2), encoding="utf-8")
        return destination


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
