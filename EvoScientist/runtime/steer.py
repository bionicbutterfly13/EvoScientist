"""Process-runtime steer channel for EvoScientist.

Holds user "steer" messages pushed into an in-flight run, keyed by ``thread_id``.
:class:`~EvoScientist.middleware.steer.SteerMiddleware` drains them in its
``before_model`` hook — i.e. at each model-call boundary — and injects them as
``HumanMessage``\\s, so a steer slips into the ongoing turn and guides the next
model call *without interrupting* the current step.

Backend is pluggable:

- The default **in-process** backend (a module-level dict) works when the agent
  runs in the same process as the pusher — the terminal TUI.
- The **WebUI/deploy** path runs the agent in a separate ``langgraph dev``
  subprocess, so a process-local dict cannot cross that boundary. A shared-store
  backend (file / db / LangGraph Store) must be registered there via
  :func:`set_steer_backend`. That wiring is Part B2.
"""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable

# Explicit steer gesture in the TUI: a line beginning with this token is always
# treated as a steer, regardless of ``steer_mode``.
STEER_PREFIX = "/steer"


@runtime_checkable
class SteerBackend(Protocol):
    """Storage for pending steer messages keyed by thread id."""

    def push(self, thread_id: str, text: str) -> None: ...

    def drain(self, thread_id: str) -> list[str]: ...


class InProcessSteerBackend:
    """Default backend: an in-memory, thread-safe ``{thread_id: [text, ...]}``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, list[str]] = {}

    def push(self, thread_id: str, text: str) -> None:
        text = (text or "").strip()
        if not thread_id or not text:
            return
        with self._lock:
            self._pending.setdefault(thread_id, []).append(text)

    def drain(self, thread_id: str) -> list[str]:
        if not thread_id:
            return []
        with self._lock:
            return self._pending.pop(thread_id, [])


_backend: SteerBackend = InProcessSteerBackend()


def set_steer_backend(backend: SteerBackend) -> None:
    """Swap the steer backend (e.g. a shared store for the WebUI subprocess)."""
    global _backend
    _backend = backend


def get_steer_backend() -> SteerBackend:
    """Return the active steer backend."""
    return _backend


def push_steer(thread_id: str, text: str) -> None:
    """Queue *text* as a steer for *thread_id* (no-op on empty inputs)."""
    if thread_id and text and text.strip():
        _backend.push(thread_id, text)


def drain_steer(thread_id: str) -> list[str]:
    """Return and clear all pending steer messages for *thread_id*."""
    if not thread_id:
        return []
    return _backend.drain(thread_id)


def has_pending_steer(thread_id: str) -> bool:
    """Whether *thread_id* has steer messages queued (non-destructive)."""
    backend = _backend
    if isinstance(backend, InProcessSteerBackend):
        with backend._lock:
            return bool(backend._pending.get(thread_id))
    # Unknown backend: a destructive drain-and-check would lose data, so report
    # False rather than consume the queue.
    return False


def parse_steer_command(text: str) -> str | None:
    """Parse a ``/steer ...`` line.

    Returns the steer body (possibly ``""`` for a bare ``/steer``) when *text*
    is a steer command, else ``None``. Case-insensitive on the prefix.
    """
    stripped = (text or "").strip()
    lowered = stripped.lower()
    if lowered == STEER_PREFIX:
        return ""
    if lowered.startswith(STEER_PREFIX + " "):
        return stripped[len(STEER_PREFIX) :].strip()
    return None
