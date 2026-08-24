"""
Workaround for a DeepEval bug: create_api_trace() in deepeval/evaluate/utils.py
requires trace.end_time to be set, but sync evals_iterator() runs
(AsyncConfig(run_async=False)) can leave it None — tracing.py's sibling
function already has a `trace.end_time or perf_counter()` fallback for
this exact case; evaluate/utils.py's create_api_trace() is missing it.
Remove this once upstream ships the same fallback.

NOTE: `import deepeval.evaluate.execute.loop` (dotted form) breaks — the
deepeval package re-exports the evaluate() function as the `evaluate`
attribute on the top-level `deepeval` package, shadowing the `evaluate`
submodule at that attribute path. importlib.import_module goes through
the import system instead of attribute access, so it isn't affected.
"""
import importlib
from time import perf_counter

from deepeval.evaluate.utils import create_api_trace as _original_create_api_trace

_loop = importlib.import_module("deepeval.evaluate.execute.loop")

def _patched_create_api_trace(trace, golden):
    if trace.end_time is None:
        from deepeval.tracing.tracing import trace_manager
        leaked = [
            (s.name, s.uuid, s.parent_uuid)
            for s in trace_manager.active_spans.values()
            if s.trace_uuid == trace.uuid
        ]
        if leaked:
            print(
                f"[elyaeval] end_time was None for golden={golden.input!r}; "
                f"still-active spans at that moment: {leaked}"
            )
        trace.end_time = perf_counter()
    return _original_create_api_trace(trace, golden)

_loop.create_api_trace = _patched_create_api_trace