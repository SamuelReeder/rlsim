"""Bounded, cancellable local SkinTokens jobs. No user-selected executable paths."""
from __future__ import annotations

import shutil
import re
import threading
import time
import uuid
from pathlib import Path

from .rig_prepare import prepare_rig


class RigJobs:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.jobs = {}

    def reserve(self):
        with self.lock:
            if any(j['status'] in {'uploading', 'checking', 'rigging', 'verifying', 'stopping'} for j in self.jobs.values()):
                raise ValueError('A rig preparation is already running. Wait for it or cancel it first.')
            # Keep a small local result history; never delete caller-owned files.
            self.root.mkdir(parents=True, exist_ok=True)
            folders=sorted((p for p in self.root.iterdir() if p.is_dir() and not p.is_symlink()
                            and re.fullmatch(r'[a-f0-9]{32}',p.name)),key=lambda p:p.stat().st_mtime)
            while len(folders)>=8:
                folder=folders.pop(0)
                self.jobs.pop(folder.name,None)
                shutil.rmtree(folder)
            token = uuid.uuid4().hex
            (self.root / token).mkdir()
            self.jobs[token] = {'id': token, 'status': 'uploading', 'created': time.time(),
                                'cancel': threading.Event(), 'error': None}
            return token

    def snapshot(self, token):
        with self.lock:
            job = self.jobs[token]
            return {k: v for k, v in job.items() if k not in {'cancel', 'thread'}}

    def start(self, token):
        with self.lock:
            job = self.jobs[token]
            job['status'] = 'checking'
            thread = threading.Thread(target=self._run, args=(token,), daemon=True)
            job['thread'] = thread
            thread.start()
        return self.snapshot(token)

    def _run(self, token):
        job, folder = self.jobs[token], self.root / token
        def stage(value):
            with self.lock:
                if not job['cancel'].is_set():
                    job['status'] = value
        try:
            result = prepare_rig(folder / 'input.glb', folder / 'rigged.glb',
                                 preserve_mesh=True, cancel=job['cancel'], on_stage=stage)
            with self.lock:
                job.update(status='cancelled' if job['cancel'].is_set() else 'completed',
                           fidelity=result.get('fidelity'), elapsed=result['elapsed'],
                           warnings=result['warnings'])
        except Exception as exc:
            with self.lock:
                job.update(status='cancelled' if job['cancel'].is_set() else 'failed', error=str(exc))

    def stop(self, token):
        with self.lock:
            job = self.jobs[token]
            if job['status'] in {'uploading', 'checking', 'rigging', 'verifying', 'stopping'}:
                job['cancel'].set()
                job['status'] = 'stopping'
        return self.snapshot(token)

    def close(self):
        with self.lock:
            jobs = list(self.jobs.values())
            for job in jobs:
                job['cancel'].set()
        for job in jobs:
            if job.get('thread'):
                job['thread'].join(timeout=8)
