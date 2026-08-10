# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Coordinate quant cache generation across threads and ComfyUI processes."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOCK_POLL_SECONDS = 0.2
LOCK_TIMEOUT_SECONDS = 60 * 60
STALE_LOCK_SECONDS = 2 * 60 * 60


class QuantCacheBuildCoordinator:
    """Serialize builders for the same derived checkpoint identity."""

    _thread_locks: dict[str, threading.Lock] = {}
    _thread_locks_guard = threading.Lock()

    def __init__(self, lock_directory: Path) -> None:
        """Create a coordinator rooted in the global cache directory."""

        self._lock_directory = lock_directory

    @contextmanager
    def acquire(self, stable_key: str) -> Iterator[None]:
        """Acquire in-process and cross-process ownership for one cache key."""

        thread_lock = self._thread_lock(stable_key)
        with thread_lock:
            self._lock_directory.mkdir(parents=True, exist_ok=True)
            lock_path = self._lock_directory / f"{stable_key}.lock"
            self._acquire_file_lock(lock_path)
            try:
                yield
            finally:
                try:
                    lock_path.unlink(missing_ok=True)
                except OSError:
                    pass

    @classmethod
    def _thread_lock(cls, stable_key: str) -> threading.Lock:
        """Return the shared in-process lock for a stable cache key."""

        with cls._thread_locks_guard:
            return cls._thread_locks.setdefault(stable_key, threading.Lock())

    @staticmethod
    def _acquire_file_lock(lock_path: Path) -> None:
        """Create an exclusive lock file, recovering only demonstrably stale locks."""

        started = time.monotonic()
        while True:
            try:
                descriptor = os.open(
                    lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                with os.fdopen(descriptor, "w", encoding="utf-8") as lock_file:
                    lock_file.write(f"pid={os.getpid()}\ncreated={time.time()}\n")
                return
            except FileExistsError:
                try:
                    age = time.time() - lock_path.stat().st_mtime
                    if age > STALE_LOCK_SECONDS and not _lock_owner_is_alive(lock_path):
                        lock_path.unlink()
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() - started >= LOCK_TIMEOUT_SECONDS:
                    raise TimeoutError(
                        "Timed out waiting for another SimpleSyrup process to finish "
                        "the same quantized checkpoint."
                    ) from None
                time.sleep(LOCK_POLL_SECONDS)


def _lock_owner_is_alive(lock_path: Path) -> bool:
    """Return whether a well-formed lock still belongs to a running process."""

    try:
        first_line = lock_path.read_text(encoding="utf-8").splitlines()[0]
        process_id = int(first_line.removeprefix("pid="))
        os.kill(process_id, 0)
    except PermissionError:
        return True
    except (IndexError, OSError, ValueError):
        return False
    return True
