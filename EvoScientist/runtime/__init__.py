"""Process-runtime state for EvoScientist.

Small, module-level runtime values that live for the lifetime of the process
and are read/written across the agent, its middleware, and the CLI command
layer — analogous to the stream cancel-scope in ``stream/display.py``. For
example, the interaction-mode (``initiative``) level.
"""
