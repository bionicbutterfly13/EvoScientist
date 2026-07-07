"""Initiative overlay middleware for EvoScientist.

Appends a per-turn interaction-mode directive to the system prompt based on the
resolved initiative level (see ``runtime/initiative.py``). ``high`` appends
nothing; the default ``medium`` overlay keeps next-step suggestions restrained.

Mirrors :class:`~EvoScientist.middleware.runtime_context.RuntimeContextMiddleware`:
a thin ``modify_request`` that reads live state and appends to the system
message via ``append_to_system_message``. The level is resolved at request time
(not construction time), so ``/initiative`` takes effect on the next model call
without rebuilding the agent.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

from ..runtime.initiative import initiative_directive, resolve_initiative_level
from .utils import append_to_system_message


class InitiativeMiddleware(AgentMiddleware):
    """Inject an interaction-mode directive scaled to the initiative level."""

    def __init__(self, *, default_level: str = "medium") -> None:
        self._default_level = default_level

    def _directive(self) -> str:
        level = resolve_initiative_level(self._default_level)
        return initiative_directive(level)

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        """Append the initiative directive (if any) to the system prompt."""
        directive = self._directive()
        if not directive:
            return request
        new_system = append_to_system_message(request.system_message, directive)
        return request.override(system_message=new_system)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject the directive before the sync model handler."""
        return handler(self.modify_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Inject the directive before the async model handler."""
        return await handler(self.modify_request(request))


def create_initiative_middleware(
    *, default_level: str = "medium"
) -> InitiativeMiddleware:
    """Build initiative overlay middleware with the given default level."""
    return InitiativeMiddleware(default_level=default_level)
