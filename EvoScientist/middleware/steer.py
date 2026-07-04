"""Steer middleware — slip user guidance into an in-flight run.

Implements the ``before_model`` hook, which runs before every model call in the
agent loop. At each boundary it drains any steer messages queued for the active
thread (see ``runtime/steer.py``) and returns them as a state update. Because
``AgentState.messages`` uses the ``add_messages`` reducer, the injected
``HumanMessage`` is appended durably to history and the model sees it on the
very next call — the steer slips in without interrupting the current step.

Reading the thread id: uses ``langgraph.config.get_config()`` (the documented
way to reach the run ``RunnableConfig`` from inside a node/middleware; the
gateway and CLI both set ``configurable.thread_id`` on the run), matching
``configurable_model`` / ``runtime.initiative``.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage

from ..runtime.steer import drain_steer

# Framing so the model treats the injected text as live user guidance for the
# work already in progress, not a brand-new request.
_STEER_PREAMBLE = (
    "[Steer — mid-turn guidance from the user. Adjust the work you are currently "
    "doing to follow this now; do not restart from scratch.]\n"
)


def _active_thread_id() -> str | None:
    """Return the active run's ``thread_id`` from the LangGraph run config."""
    try:
        from langgraph.config import get_config

        cfg = get_config()
    except Exception:
        return None
    if not isinstance(cfg, dict):
        return None
    configurable = cfg.get("configurable") or {}
    if not isinstance(configurable, dict):
        return None
    tid = configurable.get("thread_id")
    return tid if isinstance(tid, str) and tid else None


class SteerMiddleware(AgentMiddleware):
    """Inject queued steer messages as HumanMessages before each model call."""

    def _drain_messages(self) -> list[HumanMessage]:
        thread_id = _active_thread_id()
        if not thread_id:
            return []
        return [
            HumanMessage(content=f"{_STEER_PREAMBLE}{text}")
            for text in drain_steer(thread_id)
            if text.strip()
        ]

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = self._drain_messages()
        return {"messages": messages} if messages else None

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = self._drain_messages()
        return {"messages": messages} if messages else None


def create_steer_middleware() -> SteerMiddleware:
    """Build steer middleware."""
    return SteerMiddleware()
