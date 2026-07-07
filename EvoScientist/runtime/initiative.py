"""Runtime interaction-mode (``initiative``) state for EvoScientist.

The initiative *level* controls how proactive and how verbose the main agent is
on a given turn:

- ``low``    — thought-partner mode: answer only what was asked, no next-step
  proposals, no memory narration, no unprompted tools/subagents.
- ``medium`` — answer first, at most one clearly-optional next step.
- ``high``   — today's behavior (proactive scientist); no overlay is added.

This is a per-process runtime value (like the stream cancel-scope in
``stream/display.py``). The ``/initiative`` command mutates it live; both
:class:`~EvoScientist.middleware.initiative.InitiativeMiddleware` and
:class:`~EvoScientist.middleware.memory.EvoMemoryMiddleware` read it at request
time so a change takes effect on the very next model call — no agent rebuild.

Resolution order (highest priority first):

1. ``RunnableConfig.configurable["initiative"]`` — a per-run override injected
   by a surface that sets it, read via ``langgraph.config.get_config()`` (the
   documented way to reach the run config from inside a middleware; see
   ``middleware/configurable_model.py``). This is what lets a subprocess-hosted
   agent (WebUI/deploy) receive a level it cannot read from this process's
   globals.
2. The process-global level set by ``/initiative`` this session.
3. The config-file default (``default_initiative``), passed in by the caller.
4. :data:`DEFAULT_LEVEL` as a final floor.
"""

from __future__ import annotations

LEVELS: tuple[str, ...] = ("low", "medium", "high")
DEFAULT_LEVEL = "medium"

# Process-global level set by `/initiative`. ``None`` means "unset — fall back
# to the config default". Not thread-safe by design: like the cancel-scope, it
# is a coarse per-process control, and a plain assignment is atomic enough for
# this use.
_current_level: str | None = None


def normalize_level(value: object | None) -> str | None:
    """Return a valid level string, or ``None`` if the input is not a level."""
    if value is None:
        return None
    candidate = str(value).strip().lower()
    return candidate if candidate in LEVELS else None


def set_initiative_level(level: str | None) -> None:
    """Set (or clear, with ``None``/invalid) the process-global level."""
    global _current_level
    _current_level = normalize_level(level)


def get_process_initiative_level() -> str | None:
    """Return the process-global level set by ``/initiative``, or ``None``."""
    return _current_level


def _read_configurable_initiative() -> str | None:
    """Read ``configurable.initiative`` from the active run config, if any.

    Mirrors ``configurable_model._read_model_override``: uses
    ``langgraph.config.get_config()`` and degrades to ``None`` outside a
    runnable context (e.g. in unit tests) or when the key is absent.
    """
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
    return normalize_level(configurable.get("initiative"))


def resolve_initiative_level(default: str | None = None) -> str:
    """Resolve the effective initiative level for the current turn.

    Applies the priority order documented in the module docstring.
    """
    return (
        _read_configurable_initiative()
        or get_process_initiative_level()
        or normalize_level(default)
        or DEFAULT_LEVEL
    )


_LOW_DIRECTIVE = """<interaction_mode level="low">
Default to thought-partner mode for exploratory conversation.
- Answer only what was asked; do not volunteer next steps, follow-ups, plans, or
  topic shifts, and do not summarize the user's point back as a task list.
- Take no unrequested actions.
- When the user explicitly asks you to do work (run, build, fix, implement,
  inspect, search, analyze, verify, or delegate), complete that work using the
  appropriate tools, memory preflight, and sub-agents.
- Restraint applies to UNSOLICITED work only. It does not forbid tools or
  delegation needed for the user's requested task.
- Do NOT narrate memory reads or writes.
Keep the response brief once the requested work is complete.
</interaction_mode>"""

_MEDIUM_DIRECTIVE = """<interaction_mode level="medium">
Answer the question directly first. You may add at most one brief, clearly
optional next step when it is genuinely useful. Do not narrate memory activity
and do not restructure the user's thread.
</interaction_mode>"""


def initiative_directive(level: str) -> str:
    """Return the system-prompt overlay for *level*.

    ``high`` (and any unknown level) returns ``""`` — installing the overlay is
    then a no-op, so today's behavior is unchanged until a lower level is set.
    """
    if level == "low":
        return _LOW_DIRECTIVE
    if level == "medium":
        return _MEDIUM_DIRECTIVE
    return ""


def narration_allowed(level: str) -> bool:
    """Whether the agent should verbalize memory-preflight results this turn.

    Only ``high`` keeps the "mention the observation result briefly"
    instruction; ``low``/``medium`` suppress it to cut memory chatter.
    """
    return level == "high"
