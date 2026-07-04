"""Tests for the interaction-mode / initiative feature.

Covers:
- ``EvoScientist.runtime.initiative`` — level normalization, live process-global
  state, and the resolution priority (configurable > process-global > default).
- ``EvoScientist.middleware.initiative.InitiativeMiddleware`` — no-op at ``high``,
  system-prompt overlay at ``low``/``medium``, sync/async parity.
- ``EvoMemoryMiddleware`` narration gating driven by the resolved level.
- The ``/initiative`` command — registration, live set, validation, ``--save``.
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import SystemMessage

from EvoScientist.runtime import initiative as rt
from EvoScientist.runtime.initiative import (
    DEFAULT_LEVEL,
    initiative_directive,
    narration_allowed,
    normalize_level,
    resolve_initiative_level,
    set_initiative_level,
)
from tests.conftest import run_async as _run


@pytest.fixture(autouse=True)
def _reset_process_level():
    """Isolate the process-global level between tests."""
    set_initiative_level(None)
    yield
    set_initiative_level(None)


def _set_configurable_initiative(value):
    """Return a (set, reset) pair driving the real LangGraph config contextvar."""
    from langchain_core.runnables.config import var_child_runnable_config

    token = var_child_runnable_config.set({"configurable": {"initiative": value}})
    return var_child_runnable_config, token


# =============================================================================
# 1. runtime/initiative resolution
# =============================================================================


class TestNormalize:
    def test_valid_levels(self):
        assert normalize_level("low") == "low"
        assert normalize_level(" HIGH ") == "high"
        assert normalize_level("Medium") == "medium"

    def test_invalid_and_none(self):
        assert normalize_level(None) is None
        assert normalize_level("bogus") is None
        assert normalize_level(42) is None


class TestResolve:
    def test_default_when_nothing_set(self):
        assert resolve_initiative_level(None) == DEFAULT_LEVEL  # "high"

    def test_config_default_used(self):
        assert resolve_initiative_level("low") == "low"

    def test_process_global_overrides_config_default(self):
        set_initiative_level("medium")
        assert resolve_initiative_level("low") == "medium"

    def test_invalid_process_value_clears(self):
        set_initiative_level("nonsense")
        assert resolve_initiative_level("low") == "low"

    def test_configurable_wins_over_process_and_default(self):
        set_initiative_level("high")
        var, token = _set_configurable_initiative("low")
        try:
            assert resolve_initiative_level("high") == "low"
        finally:
            var.reset(token)

    def test_configurable_ignored_when_invalid(self):
        set_initiative_level("medium")
        var, token = _set_configurable_initiative("garbage")
        try:
            # invalid configurable falls through to the process-global value
            assert resolve_initiative_level("high") == "medium"
        finally:
            var.reset(token)


class TestDirectiveAndNarration:
    def test_high_is_noop(self):
        assert initiative_directive("high") == ""
        assert initiative_directive("unknown") == ""

    def test_low_and_medium_have_text(self):
        assert 'interaction_mode level="low"' in initiative_directive("low")
        assert 'interaction_mode level="medium"' in initiative_directive("medium")

    def test_narration_only_at_high(self):
        assert narration_allowed("high") is True
        assert narration_allowed("low") is False
        assert narration_allowed("medium") is False


# =============================================================================
# 2. InitiativeMiddleware
# =============================================================================


def _make_request(system_text: str = "BASE SYSTEM"):
    """Minimal ModelRequest stub whose ``override`` records kwargs."""
    req = MagicMock()
    req.system_message = SystemMessage(content=system_text)

    def _override(**kwargs):
        new = MagicMock()
        new.system_message = kwargs.get("system_message", req.system_message)
        return new

    req.override = MagicMock(side_effect=_override)
    return req


def _system_text(system_message: SystemMessage) -> str:
    return " ".join(
        b.get("text", "") for b in system_message.content_blocks if isinstance(b, dict)
    )


class TestInitiativeMiddleware:
    def test_high_is_passthrough(self):
        from EvoScientist.middleware.initiative import create_initiative_middleware

        mw = create_initiative_middleware(default_level="high")
        req = _make_request()
        handler = MagicMock(return_value="ok")
        result = mw.wrap_model_call(req, handler)

        assert result == "ok"
        req.override.assert_not_called()
        handler.assert_called_once_with(req)

    def test_low_appends_directive(self):
        from EvoScientist.middleware.initiative import create_initiative_middleware

        mw = create_initiative_middleware(default_level="low")
        req = _make_request()
        handler = MagicMock(return_value="ok")
        mw.wrap_model_call(req, handler)

        req.override.assert_called_once()
        new_sys = req.override.call_args.kwargs["system_message"]
        text = _system_text(new_sys)
        assert "BASE SYSTEM" in text
        assert "interaction_mode" in text
        # handler receives the OVERRIDDEN request
        assert handler.call_args[0][0] is not req

    def test_live_process_level_beats_default(self):
        from EvoScientist.middleware.initiative import create_initiative_middleware

        # built with default high, but /initiative lowered it live
        mw = create_initiative_middleware(default_level="high")
        set_initiative_level("low")
        req = _make_request()
        handler = MagicMock(return_value="ok")
        mw.wrap_model_call(req, handler)
        req.override.assert_called_once()

    def test_async_parity(self):
        from EvoScientist.middleware.initiative import create_initiative_middleware

        mw = create_initiative_middleware(default_level="medium")
        req = _make_request()

        async def handler(r):
            return r

        result = _run(mw.awrap_model_call(req, handler))
        # medium appends → handler got the overridden request
        assert result is not req
        assert "interaction_mode" in _system_text(result.system_message)


# =============================================================================
# 3. Memory narration gating
# =============================================================================


class TestMemoryNarrationGating:
    def _middleware(self):
        from EvoScientist.middleware.memory import create_memory_middleware

        return create_memory_middleware(
            tempfile.mkdtemp(),
            workspace_dir=tempfile.mkdtemp(),
            initiative_default_level="high",
        )

    def test_high_narrates(self):
        mw = self._middleware()
        set_initiative_level("high")
        assert "Mention the result briefly" in mw._observation_memory_instructions()

    def test_low_suppresses(self):
        mw = self._middleware()
        set_initiative_level("low")
        assert "Mention the result briefly" not in mw._observation_memory_instructions()

    def test_medium_suppresses(self):
        mw = self._middleware()
        set_initiative_level("medium")
        assert "Mention the result briefly" not in mw._observation_memory_instructions()


# =============================================================================
# 4. /initiative command
# =============================================================================


class TestInitiativeCommand:
    def test_registered_with_alias(self):
        import EvoScientist.commands.implementation  # noqa: F401 (registers)
        from EvoScientist.commands.manager import manager

        cmd = manager.get_command("/initiative")
        assert cmd is not None
        assert manager.get_command("/mode") is cmd

    def _ctx(self):
        ctx = MagicMock()
        ctx.ui.append_system = MagicMock()
        ctx.config = MagicMock()
        ctx.config.default_initiative = "high"
        return ctx

    def test_sets_live_level(self):
        from EvoScientist.commands.implementation.initiative import InitiativeCommand

        ctx = self._ctx()
        fake_cfg = MagicMock()
        fake_cfg.default_initiative = "high"
        with patch("EvoScientist.EvoScientist._ensure_config", return_value=fake_cfg):
            _run(InitiativeCommand().execute(ctx, ["low"]))

        assert rt.get_process_initiative_level() == "low"

    def test_rejects_invalid_level(self):
        from EvoScientist.commands.implementation.initiative import InitiativeCommand

        ctx = self._ctx()
        fake_cfg = MagicMock()
        fake_cfg.default_initiative = "high"
        with patch("EvoScientist.EvoScientist._ensure_config", return_value=fake_cfg):
            _run(InitiativeCommand().execute(ctx, ["turbo"]))

        assert rt.get_process_initiative_level() is None  # unchanged
        # an error was surfaced
        styles = [c.kwargs.get("style") for c in ctx.ui.append_system.call_args_list]
        assert "red" in styles

    def test_save_persists(self):
        from EvoScientist.commands.implementation.initiative import InitiativeCommand

        ctx = self._ctx()
        fake_cfg = MagicMock()
        fake_cfg.default_initiative = "high"
        with (
            patch("EvoScientist.EvoScientist._ensure_config", return_value=fake_cfg),
            patch(
                "EvoScientist.config.settings.set_config_value", return_value=True
            ) as mock_set,
        ):
            _run(InitiativeCommand().execute(ctx, ["medium", "--save"]))

        mock_set.assert_called_once_with("default_initiative", "medium")
        assert fake_cfg.default_initiative == "medium"
        assert rt.get_process_initiative_level() == "medium"
