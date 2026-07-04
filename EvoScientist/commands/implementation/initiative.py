"""`/initiative` — set the session's interaction mode / initiative level live.

Unlike `/model` (which rebuilds the agent), this only mutates a process-global
level that `InitiativeMiddleware` and `EvoMemoryMiddleware` read at request
time, so it takes effect on the very next reply with no rebuild.
"""

from __future__ import annotations

from typing import ClassVar

from ..base import Argument, Command, CommandContext
from ..manager import manager


class InitiativeCommand(Command):
    """Set how proactive/verbose EvoScientist is for this session."""

    name = "/initiative"
    alias: ClassVar[list[str]] = ["/mode"]
    description = "Set initiative: low | medium | high (--save to persist)"
    arguments: ClassVar[list[Argument]] = [
        Argument(
            name="level",
            type=str,
            description="low | medium | high. Shows the current level if omitted.",
            required=False,
        ),
        Argument(
            name="--save",
            type=bool,
            description="Persist as the default_initiative config value",
            required=False,
        ),
    ]

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        # Lazy imports: importing EvoScientist / config at module load time
        # would create a circular import through commands/implementation.
        from ...config.settings import set_config_value
        from ...EvoScientist import _ensure_config
        from ...runtime.initiative import (
            LEVELS,
            resolve_initiative_level,
            set_initiative_level,
        )

        cfg = _ensure_config()
        save = "--save" in args
        rest = [a for a in args if a != "--save"]

        if not rest:
            current = resolve_initiative_level(cfg.default_initiative)
            ctx.ui.append_system(
                f"Initiative: {current}. "
                f"Usage: /initiative <{' | '.join(LEVELS)}> [--save]",
                style="dim",
            )
            return

        level = rest[0].strip().lower()
        if level not in LEVELS:
            ctx.ui.append_system(
                f"Unknown level '{rest[0]}'. Choose: {', '.join(LEVELS)}.",
                style="red",
            )
            return

        set_initiative_level(level)

        saved_note = ""
        if save:
            if set_config_value("default_initiative", level):
                cfg.default_initiative = level
                saved_note = " (saved to config)"
            else:
                saved_note = " (save failed)"

        ctx.ui.append_system(f"Initiative set to {level}{saved_note}.", style="green")


manager.register(InitiativeCommand())
