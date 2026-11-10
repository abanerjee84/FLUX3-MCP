from __future__ import annotations

import json
import logging
import secrets
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from .schema import ImageRequest

log = logging.getLogger(__name__)
TERMINAL = {"succeeded", "failed", "cancelled"}


class Cancelled(Exception):
    pass


class JobStore:
    """SQLite makes queue submission atomic across CLI, MCP and HTTP clients."""

    def __init__(self, home: Path, backend):
        self.home = home
        self.backend = backend
        home.mkdir(parents=True, exist_ok=True)
        (home / "outputs").mkdir(exist_ok=True)
        self.db = home / "jobs.sqlite3"
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.stop = threading.Event()
        self.current_cancel = threading.Event()
        self.current_id = None
        self.thread = None
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, request_key TEXT UNIQUE, request TEXT NOT NULL,
                status TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0, stage TEXT, error TEXT, result TEXT)""")
            db.execute("UPDATE jobs SET status='failed', error='Service restarted during inference', updated=? WHERE status='running'", (time.time(),))

    def connect(self):
        return sqlite3.connect(self.db, timeout=15)

    def start(self):
        self.thread = threading.Thread(target=self._run, name="flux-gpu-queue", daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        self.current_cancel.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=30)

    def submit(self, req: ImageRequest) -> dict:
        with self.lock, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if req.client_request_id:
                old = db.execute("SELECT id,request FROM jobs WHERE request_key=?", (req.client_request_id,)).fetchone()
                if old:
                    previous = json.loads(old[1])
                    current = req.model_dump()
                    # An automatically chosen seed is not part of retry identity.
                    if current["seed"] is None:
                        current["seed"] = previous["seed"]
                    if current != previous:
                        raise ValueError("client_request_id already belongs to a different request")
                    return self.get(old[0])
            count = db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if count >= 32:
                raise OverflowError("Queue is full (32 jobs)")
            data = req.model_dump()
            if data["seed"] is None:
                data["seed"] = secrets.randbelow(2**32)
            job_id = uuid.uuid4().hex
            now = time.time()
            db.execute("INSERT INTO jobs(id,request_key,request,status,created,updated) VALUES(?,?,?,'queued',?,?)",
                       (job_id, req.client_request_id, json.dumps(data), now, now))
        self.wake.set()
        return self.get(job_id)

    def get(self, job_id: str) -> dict:
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        result = dict(row)
        result.pop("request_key")
        result["request"] = json.loads(result["request"])
        result["result"] = json.loads(result["result"]) if result["result"] else None
        result["job_id"] = result.pop("id")
        return result

    def update(self, job_id, **fields):
        fields["updated"] = time.time()
        with self.connect() as db:
            db.execute(f"UPDATE jobs SET {','.join(k + '=?' for k in fields)} WHERE id=?", (*fields.values(), job_id))

    def cancel(self, job_id: str):
        with self.lock:
            job = self.get(job_id)
            if job["status"] == "queued":
                self.update(job_id, status="cancelled", stage="cancelled")
            elif job["status"] == "running":
                self.current_cancel.set()
                self.update(job_id, stage="cancelling")
        return self.get(job_id)

    def _run(self):
        while not self.stop.is_set():
            with self.lock, self.connect() as db:
                row = db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
                if row:
                    self.current_id = row[0]
                    self.current_cancel.clear()
                    self.update(row[0], status="running", stage="loading")
            if not row:
                self.wake.wait(0.5)
                self.wake.clear()
                continue
            job_id = row[0]
            try:
                req = ImageRequest.model_validate(self.get(job_id)["request"])
                def progress(stage, percent, job_id=job_id):
                    if self.current_cancel.is_set():
                        raise Cancelled()
                    self.update(job_id, stage=stage, progress=percent)
                result = self.backend.generate(req, job_id, progress, self.current_cancel)
                if self.current_cancel.is_set():
                    raise Cancelled()
                self.update(job_id, status="succeeded", stage="done", progress=100, result=json.dumps(result))
            except Cancelled:
                self.update(job_id, status="cancelled", stage="cancelled")
            except Exception as exc:
                log.exception("Image job %s failed", job_id)
                self.update(job_id, status="failed", stage="failed", error=f"{type(exc).__name__}: {exc}")
            finally:
                with self.lock:
                    self.current_id = None
