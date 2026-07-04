"""Tests for slip-in steering.

Covers:
- ``EvoScientist.runtime.steer`` — the thread-keyed steer channel, the pluggable
  backend, and ``/steer`` command parsing.
- ``EvoScientist.middleware.steer.SteerMiddleware`` — its ``before_model`` hook
  drains the channel for the active thread (read from the LangGraph run config)
  and returns a ``messages`` state update, or ``None`` when nothing is queued.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from EvoScientist.runtime.steer import (
    InProcessSteerBackend,
    drain_steer,
    has_pending_steer,
    parse_steer_command,
    push_steer,
    set_steer_backend,
)
from tests.conftest import run_async as _run


@pytest.fixture(autouse=True)
def _fresh_backend():
    """Isolate the steer channel between tests."""
    set_steer_backend(InProcessSteerBackend())
    yield
    set_steer_backend(InProcessSteerBackend())


def _set_thread(thread_id: str | None):
    """Drive the real LangGraph config contextvar; returns (var, token)."""
    from langchain_core.runnables.config import var_child_runnable_config

    value = {"configurable": {"thread_id": thread_id}} if thread_id else {}
    token = var_child_runnable_config.set(value)
    return var_child_runnable_config, token


# =============================================================================
# 1. Steer channel
# =============================================================================


class TestChannel:
    def test_push_then_drain_is_fifo(self):
        push_steer("t1", "first")
        push_steer("t1", "second")
        assert drain_steer("t1") == ["first", "second"]

    def test_drain_clears(self):
        push_steer("t1", "x")
        assert drain_steer("t1") == ["x"]
        assert drain_steer("t1") == []

    def test_thread_isolation(self):
        push_steer("t1", "a")
        push_steer("t2", "b")
        assert drain_steer("t2") == ["b"]
        assert drain_steer("t1") == ["a"]

    def test_empty_and_blank_ignored(self):
        push_steer("t1", "")
        push_steer("t1", "   ")
        push_steer("", "orphan")
        assert drain_steer("t1") == []

    def test_has_pending_is_nondestructive(self):
        push_steer("t1", "a")
        assert has_pending_steer("t1") is True
        assert has_pending_steer("t1") is True  # still there
        assert drain_steer("t1") == ["a"]
        assert has_pending_steer("t1") is False

    def test_backend_swap(self):
        push_steer("t1", "old")
        set_steer_backend(InProcessSteerBackend())
        # new backend has nothing
        assert drain_steer("t1") == []


class TestParseSteerCommand:
    def test_prefixed(self):
        assert parse_steer_command("/steer just answer") == "just answer"

    def test_case_insensitive(self):
        assert parse_steer_command("/STEER Stop") == "Stop"

    def test_bare_prefix_returns_empty(self):
        assert parse_steer_command("/steer") == ""
        assert parse_steer_command("  /steer  ") == ""

    def test_non_steer_returns_none(self):
        assert parse_steer_command("hello") is None
        assert parse_steer_command("/steering is fun") is None
        assert parse_steer_command("") is None


# =============================================================================
# 2. SteerMiddleware.before_model
# =============================================================================


class TestSteerMiddleware:
    def _mw(self):
        from EvoScientist.middleware.steer import create_steer_middleware

        return create_steer_middleware()

    def test_injects_pending_as_messages(self):
        mw = self._mw()
        push_steer("run-1", "stop rambling, just answer")
        var, token = _set_thread("run-1")
        try:
            update = mw.before_model({"messages": []}, None)
        finally:
            var.reset(token)

        assert update is not None
        msgs = update["messages"]
        assert len(msgs) == 1
        assert isinstance(msgs[0], HumanMessage)
        assert "stop rambling, just answer" in msgs[0].content
        assert "Steer" in msgs[0].content  # preamble present
        # drained: a second call finds nothing
        var, token = _set_thread("run-1")
        try:
            assert mw.before_model({"messages": []}, None) is None
        finally:
            var.reset(token)

    def test_no_pending_returns_none(self):
        mw = self._mw()
        var, token = _set_thread("run-1")
        try:
            assert mw.before_model({"messages": []}, None) is None
        finally:
            var.reset(token)

    def test_no_thread_returns_none(self):
        mw = self._mw()
        push_steer("run-1", "orphan")  # different / no thread in context
        var, token = _set_thread(None)
        try:
            assert mw.before_model({"messages": []}, None) is None
        finally:
            var.reset(token)
        # still queued for its real thread
        assert drain_steer("run-1") == ["orphan"]

    def test_async_parity(self):
        mw = self._mw()
        push_steer("run-2", "focus")
        var, token = _set_thread("run-2")
        try:
            update = _run(mw.abefore_model({"messages": []}, None))
        finally:
            var.reset(token)
        assert update is not None
        assert "focus" in update["messages"][0].content
