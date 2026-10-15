"""Structured wall-clock timing for SAGE pipeline research runs.

The emitted JSON lines are deliberately independent of human-readable logs so
post-processing does not have to infer stages or object counts with regexes.
"""
from __future__ import annotations
import contextvars
import functools
import inspect
import json
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
TIMING_PREFIX = 'SAGE_TIMING_EVENT '

@dataclass
class _ToolCallState:
    call_id: str
    tool_name: str
    started_ns: int
    sequence: int = 0
    operation: Optional[str] = None
    requested_count: Optional[int] = None
    attempted_count: Optional[int] = None
    succeeded_count: Optional[int] = None
    count_details: Dict[str, Any] = field(default_factory=dict)
_current_call: contextvars.ContextVar[Optional[_ToolCallState]] = contextvars.ContextVar('sage_timing_call', default=None)
_stage_stack: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar('sage_timing_stage_stack', default=())
_print_lock = threading.Lock()

def _emit(payload: Dict[str, Any]) -> None:
    if os.environ.get('SCENEPACE_STAGE_TIMING', '1') == '0':
        return
    payload = {'pid': os.getpid(), **payload}
    line = TIMING_PREFIX + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    with _print_lock:
        print(line, file=sys.stderr, flush=True)

def _next_sequence(state: _ToolCallState) -> int:
    state.sequence += 1
    return state.sequence

class StageTimer:
    """Measure one inclusive stage with ``perf_counter_ns`` wall time."""

    def __init__(self, stage: str, category: str, *, aggregation: str='wall', details: Optional[Dict[str, Any]]=None) -> None:
        if aggregation not in {'wall', 'cumulative'}:
            raise ValueError("aggregation must be 'wall' or 'cumulative'")
        self.stage = stage
        self.category = category
        self.aggregation = aggregation
        self.details = dict(details or {})
        self._state: Optional[_ToolCallState] = None
        self._event_id: Optional[str] = None
        self._parent_event_id: Optional[str] = None
        self._started_ns: Optional[int] = None
        self._stack_token = None

    def add_details(self, **details: Any) -> None:
        self.details.update(details)

    def __enter__(self) -> 'StageTimer':
        self._state = _current_call.get()
        if self._state is None:
            return self
        stack = _stage_stack.get()
        self._event_id = uuid.uuid4().hex
        self._parent_event_id = stack[-1] if stack else None
        self._started_ns = time.perf_counter_ns()
        self._stack_token = _stage_stack.set((*stack, self._event_id))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self._state is None or self._started_ns is None:
            return False
        ended_ns = time.perf_counter_ns()
        if self._stack_token is not None:
            _stage_stack.reset(self._stack_token)
        payload: Dict[str, Any] = {'event': 'stage', 'call_id': self._state.call_id, 'tool_name': self._state.tool_name, 'sequence': _next_sequence(self._state), 'event_id': self._event_id, 'parent_event_id': self._parent_event_id, 'stage': self.stage, 'category': self.category, 'aggregation': self.aggregation, 'clock': 'perf_counter_ns', 'started_ns': self._started_ns, 'ended_ns': ended_ns, 'duration_s': round((ended_ns - self._started_ns) / 1000000000, 6), 'status': 'error' if exc_type else 'ok'}
        if self.details:
            payload['details'] = self.details
        if exc_type:
            payload['error_type'] = exc_type.__name__
        _emit(payload)
        return False

def timed_stage(stage: str, category: str, *, aggregation: str='wall', details: Optional[Dict[str, Any]]=None) -> StageTimer:
    return StageTimer(stage, category, aggregation=aggregation, details=details)

def set_operation_counts(operation: str, *, attempted: Optional[int], succeeded: Optional[int], requested: Optional[int]=None, **details: Any) -> None:
    """Attach actual operation counts to the enclosing MCP tool call."""
    state = _current_call.get()
    if state is None:
        return
    state.operation = operation
    state.attempted_count = None if attempted is None else int(attempted)
    state.succeeded_count = None if succeeded is None else int(succeeded)
    state.requested_count = None if requested is None else int(requested)
    state.count_details.update(details)

def timed_tool(tool_name: Optional[str]=None):
    """Decorate an MCP tool and emit start/end records, including early returns."""

    def decorator(function):
        name = tool_name or function.__name__

        @functools.wraps(function)
        async def async_wrapper(*args, **kwargs):
            started_ns = time.perf_counter_ns()
            state = _ToolCallState(uuid.uuid4().hex, name, started_ns)
            call_token = _current_call.set(state)
            stack_token = _stage_stack.set(())
            _emit({'event': 'tool_start', 'call_id': state.call_id, 'tool_name': name, 'started_ns': started_ns, 'clock': 'perf_counter_ns'})
            error_type = None
            try:
                return await function(*args, **kwargs)
            except BaseException as exc:
                error_type = type(exc).__name__
                raise
            finally:
                ended_ns = time.perf_counter_ns()
                payload: Dict[str, Any] = {'event': 'tool_end', 'call_id': state.call_id, 'tool_name': name, 'sequence': _next_sequence(state), 'started_ns': started_ns, 'ended_ns': ended_ns, 'duration_s': round((ended_ns - started_ns) / 1000000000, 6), 'clock': 'perf_counter_ns', 'status': 'error' if error_type else 'completed', 'operation': state.operation, 'requested_count': state.requested_count, 'attempted_count': state.attempted_count, 'succeeded_count': state.succeeded_count}
                if state.count_details:
                    payload['count_details'] = state.count_details
                if error_type:
                    payload['error_type'] = error_type
                _emit(payload)
                _stage_stack.reset(stack_token)
                _current_call.reset(call_token)

        @functools.wraps(function)
        def sync_wrapper(*args, **kwargs):
            raise TypeError('timed_tool currently supports async MCP tools only')
        return async_wrapper if inspect.iscoroutinefunction(function) else sync_wrapper
    return decorator
