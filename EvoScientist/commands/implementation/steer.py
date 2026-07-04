"""`/steer` — slip guidance into the running turn without interrupting it.

In the TUI, a ``/steer`` typed while the agent is busy is intercepted by the
input handler (before command dispatch) and pushed onto the steer channel, so
this command object handles the non-busy / non-TUI case: it pushes onto the
active thread's channel, where ``SteerMiddleware`` drains it at the next
model-call boundary.
"""

from __future__ import annotations

from typing import ClassVar

from ..base import Argument, Command, CommandContext
from ..manager import manager


class SteerCommand(Command):
    """Inject mid-run guidance into the active conversation's next model call."""

    name = "/steer"
    description = "Slip guidance into the running turn (mid-run steer)"
    arguments: ClassVar[list[Argument]] = [
        Argument(
            name="text",
            type=str,
            description="Guidance to slip into the running turn",
            required=False,
        ),
    ]

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        from ...runtime.steer import push_steer

        body = " ".join(args).strip()
        if not body:
            ctx.ui.append_system(
                "Usage: /steer <guidance to slip into the running turn>",
                style="yellow",
            )
            return

        thread_id = getattr(ctx, "thread_id", None)
        if not thread_id:
            ctx.ui.append_system("No active conversation to steer.", style="yellow")
            return

        push_steer(thread_id, body)
        ctx.ui.append_system(f"↳ steering: {body}", style="cyan")


manager.register(SteerCommand())
