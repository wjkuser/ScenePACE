"""AReaL-compatible SAGE room generation agent."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from openai import APIConnectionError, InternalServerError, RateLimitError

PROJECT_ROOT = Path(os.environ.get("SAGE_EFFICIENCY_ROOT", str(Path(__file__).resolve().parents[2]))).resolve()
SERVER_DIR = PROJECT_ROOT / "server"
DEFAULT_SERVER_PATH = SERVER_DIR / "layout_wo_robot.py"
DEFAULT_SERVER_KEY_FILE = SERVER_DIR / "key.json"
DEFAULT_RL_MODEL_PATH = "models/Qwen3-14B"
DEFAULT_SERVER_MODEL_PATH = "models/Qwen3.6-35B-A3B"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from client.client_generation_room_desc import (  # noqa: E402
    MCPClientOAI,
    ContextBudgetExhausted,
    get_task_definition_text,
)
from sage_rl.policy import (TEMPERATURE, TOP_P, TOP_K, STOP_TOKEN_IDS,
                            MAX_CONTEXT_TOKENS, MAX_OUTPUT_TOKENS, sampling_contract)
from sage_rl.failure import classify_outcome
from sage_rl.task_schema import validate_task
from sage_rl.reward import (  # noqa: E402
    compute_episode_reward,
    load_layout_data,
)


_ISAAC_INFRASTRUCTURE_ERROR_MARKERS = (
    "remote isaac transport failed",
    "remote isaac job failed",
    "communication error with isaac",
    "isaac scene creation failed",
    "isaac simulation failed",
    "could not connect to isaac",
    "not connected to isaac",
    "connection refused",
)


def tool_names_from_messages(messages: List[Dict[str, Any]]) -> List[str]:
    """Return one stable tool label per assistant completion in message order."""
    labels: List[str] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls") or []
        names: List[str] = []
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
            name = function.get("name") if isinstance(function, dict) else getattr(function, "name", None)
            if name:
                names.append(str(name))
        if not names:
            content = str(message.get("content") or "")
            for block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, flags=re.DOTALL):
                try:
                    parsed = json.loads(block)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict) and parsed.get("name"):
                    names.append(str(parsed["name"]))
        labels.append("+".join(sorted(set(names))) if names else "unknown")
    return labels


def layout_object_count(layout_data: Any) -> int:
    """Count persisted scene objects, or return -1 when no layout was saved."""
    if not isinstance(layout_data, dict):
        return -1
    return sum(
        len(room.get("objects") or [])
        for room in layout_data.get("rooms") or []
        if isinstance(room, dict)
    )


def rollout_infrastructure_failure(messages: List[Dict[str, Any]]) -> str:
    """Return a failed-tool infrastructure error that must not train the policy."""
    for message in messages:
        if message.get("role") != "tool":
            continue
        content = str(message.get("content", ""))
        payload: Dict[str, Any] = {}
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                payload = parsed
        except json.JSONDecodeError:
            pass
        failed = bool(message.get("error")) or payload.get("success") is False
        lowered = content.lower()
        if failed and any(marker in lowered for marker in _ISAAC_INFRASTRUCTURE_ERROR_MARKERS):
            return next(
                marker
                for marker in _ISAAC_INFRASTRUCTURE_ERROR_MARKERS
                if marker in lowered
            )
    return ""


def normalize_server_path(path: str) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if candidate.is_file():
        return str(candidate)
    fallback = DEFAULT_SERVER_PATH.resolve()
    if fallback.is_file():
        return str(fallback)
    return str(candidate)

_ORIGINAL_PROCESS_QUERY = MCPClientOAI.process_query



def _prepare_rl_completion_kwargs(
    kwargs: Dict[str, Any],
    max_output_tokens: int,
) -> Dict[str, Any]:
    """Apply RL-only generation controls without mutating caller-owned data."""
    prepared = dict(kwargs)
    # Efficiency retains display metadata in its history. AReaL's conversation
    # cache compares wire messages, so logging fields must stay out of requests.
    log_fields = {"timestamp", "iteration", "has_tool_calls", "images_metadata",
                  "content_for_api"}
    if "messages" in prepared:
        prepared["messages"] = [
            {key: value for key, value in message.items() if key not in log_fields}
            if isinstance(message, dict) else message
            for message in prepared["messages"]
        ]
    requested = int(prepared.get("max_tokens", max_output_tokens))
    prepared["max_tokens"] = min(requested, max_output_tokens)
    prepared["top_p"] = TOP_P
    prepared["frequency_penalty"] = 0.0
    prepared["presence_penalty"] = 0.0

    extra_body = dict(prepared.get("extra_body") or {})
    extra_body.update(top_k=TOP_K, stop_token_ids=STOP_TOKEN_IDS, repetition_penalty=1.0)
    chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
    chat_template_kwargs["enable_thinking"] = False
    extra_body["chat_template_kwargs"] = chat_template_kwargs
    logit_bias = dict(extra_body.get("logit_bias") or {})
    # Qwen's thinking delimiters are single special tokens.  AReaL currently
    # samples through vLLM's /v1/completions endpoint, where logit_bias is the
    # supported way to block them at generation time.
    logit_bias["151667"] = -100.0  # <think>
    logit_bias["151668"] = -100.0  # </think>
    extra_body["logit_bias"] = logit_bias
    prepared["extra_body"] = extra_body
    return prepared


@lru_cache(maxsize=1)
def _policy_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        os.environ.get("SAGE_RL_TOKENIZER_PATH", DEFAULT_RL_MODEL_PATH),
        local_files_only=True,
    )


def _fit_context_budget(prepared, tokenizer, context_limit=16384, on_compact=None):
    from sage_rl.agent.context_compaction import fit_context
    return fit_context(prepared, tokenizer, context_limit, ContextBudgetExhausted, on_compact)


def _policy_response_failure_reason(response: Any, max_output_tokens: int) -> str:
    """Identify invalid policy responses while retaining their training tokens."""
    choices = list(getattr(response, "choices", None) or [])
    if not choices:
        return ""
    choice = choices[0]
    finish_reason = str(getattr(choice, "finish_reason", "") or "").lower()
    if finish_reason in {"length", "max_tokens"}:
        return f"policy output truncated at max_tokens={max_output_tokens}"

    message = getattr(choice, "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", "") or ""
    reasoning = (
        getattr(message, "reasoning_content", None)
        or getattr(message, "reasoning", None)
        or ""
    )
    if reasoning or (isinstance(content, str) and "<think>" in content.lower()):
        return "policy response contained reasoning despite thinking being disabled"
    if isinstance(content, str) and "<tool_call>" in content:
        blocks = re.findall(
            r"<tool_call>\s*(.*?)\s*</tool_call>",
            content,
            flags=re.DOTALL,
        )
        if not blocks:
            return "policy emitted an unclosed tool_call block"
        for block in blocks:
            try:
                tool_data = json.loads(block)
            except json.JSONDecodeError as exc:
                return f"policy emitted malformed tool JSON: {exc.msg}"
            if not isinstance(tool_data, dict) or not isinstance(
                tool_data.get("arguments"), dict
            ):
                return "policy emitted a tool call without an arguments object"
    return ""


async def _cleanup_mcp_client(client: MCPClientOAI) -> None:
    timeout_s = float(os.environ.get("SAGE_RL_MCP_CLEANUP_TIMEOUT_SECONDS", "120"))
    try:
        await asyncio.wait_for(client.cleanup(), timeout=timeout_s)
    except asyncio.TimeoutError:
        print(
            f"[SageRoomAgent] MCP cleanup timed out after {timeout_s:.0f}s, continuing",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[SageRoomAgent] MCP cleanup failed ({exc!r}), continuing",
            flush=True,
        )


async def _rl_process_query(
    self: MCPClientOAI,
    query: str,
    image_paths=None,
    system_message=None,
):
    max_out = MAX_OUTPUT_TOKENS
    session_timeout = float(
        os.environ.get("SAGE_RL_SESSION_TIMEOUT_SECONDS", "1800")
    )
    # Native scene dependencies can kill an MCP process without a Python
    # exception. Detect a broken subprocess before the full episode timeout;
    # a process exit alone does not establish that the policy caused it.
    mcp_check_interval = float(
        os.environ.get("SAGE_RL_MCP_CHECK_INTERVAL", "10")
    )
    original_create = self.openai_client.chat.completions.create
    original_call_tool = self.call_tool_on_server
    self._sage_completion_ids = []
    self._sage_process_failure_reason = ""
    self._sage_environment_failure_reason = ""
    self._sage_policy_response_failure = ""
    if not hasattr(self, "policy_failure_events"):
        self.policy_failure_events = []
    self._sage_blocked_layout_regenerations = 0
    self._sage_context_compactions = []

    def record_compaction(event):
        self._sage_context_compactions.append(event)
        print("SAGE_CONTEXT_COMPACTION " + json.dumps(event), flush=True)

    def create_with_cap(**kwargs):
        self._sage_policy_response_failure = ""
        if getattr(self, "external_actor", False):
            prepared = dict(kwargs)
        else:
            prepared = _fit_context_budget(
                _prepare_rl_completion_kwargs(kwargs, max_out), _policy_tokenizer(),
                MAX_CONTEXT_TOKENS, on_compact=record_compaction,
            )
        prepared["temperature"] = self.sampling_temperature
        if self.sampling_seed is not None:
            prepared["seed"] = int(self.sampling_seed) + len(self._sage_completion_ids)
        try:
            response = original_create(**prepared)
        except (APIConnectionError, InternalServerError, RateLimitError) as exc:
            # Efficiency catches API exceptions in its retry loop. Preserve
            # transport provenance so an exhausted retry is not a no-layout
            # policy failure. A successful retry clears the transient failure.
            self._sage_environment_failure_reason = str(exc)
            raise
        self._sage_environment_failure_reason = ""
        completion_id = getattr(response, "id", None)
        if completion_id:
            self._sage_completion_ids.append(str(completion_id))
        if getattr(self, "external_actor", False):
            failure_reason = ("external actor output budget exhausted"
                              if response.choices[0].finish_reason == "length" else "")
        else:
            failure_reason = _policy_response_failure_reason(response, max_out)
        if failure_reason:
            choices = list(getattr(response, "choices", None) or [])
            message = getattr(choices[0], "message", None) if choices else None
            raw_content = getattr(message, "content", "") if message else ""
            usage = getattr(response, "usage", None)
            self._sage_policy_response_failure = failure_reason
            self.policy_failure_events.append(
                {
                    "completion_id": str(completion_id or ""),
                    "failure_reason": failure_reason,
                    "finish_reason": str(
                        getattr(choices[0], "finish_reason", "") if choices else ""
                    ),
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(
                        getattr(usage, "completion_tokens", 0) or 0
                    ),
                    "raw_content": str(raw_content or "")[:20000],
                }
            )
            print(f"[SageRoomAgent] {failure_reason}", flush=True)
            if message is not None:
                message.content = ""
                message.tool_calls = None
        return response

    self.openai_client.chat.completions.create = create_with_cap

    async def call_tool_with_layout_guard(tool_name, tool_input):
        single_layout = os.environ.get("SAGE_RL_SINGLE_LAYOUT", "1").lower() in {
            "1", "true", "yes", "on",
        }
        if single_layout and tool_name == "generate_room_layout" and self.layout_id:
            self._sage_blocked_layout_regenerations += 1
            payload = json.dumps(
                {
                    "success": False,
                    "error": (
                        "A room layout already exists. Continue furnishing the existing "
                        "room; generate_room_layout can only be called once per rollout."
                    ),
                    "layout_id": self.layout_id,
                    "rl_layout_regeneration_blocked": True,
                }
            )
            print(
                "[SageRoomAgent] blocked generate_room_layout after layout "
                f"{self.layout_id}",
                flush=True,
            )
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=payload)]
            )
        current_task = getattr(self, "_sage_task", None)
        if current_task and tool_name in {"place_objects_in_room", "move_one_object_with_condition_in_room"}:
            field = "placement_conditions" if tool_name == "place_objects_in_room" else "condition"
            tool_input = dict(tool_input)
            tool_input[field] = str(tool_input.get(field, "")) + "\nRoom requirements (apply only the requested correction/addition; do not regenerate existing items): " + current_task["room_desc"]
        result = await original_call_tool(tool_name, tool_input)
        if current_task:
            for item in getattr(result, "content", []):
                if getattr(item, "type", None) != "text":
                    continue
                try:
                    payload = json.loads(item.text)
                except (ValueError, TypeError):
                    continue
                if not isinstance(payload, dict):
                    continue
                layout_id = payload.get("layout_id") or self.layout_id
                if layout_id:
                    scene_file = Path(os.environ["SAGE_RESULTS_DIR"]) / layout_id / f"{layout_id}.json"
                    if scene_file.is_file():
                        from sage_rl.evaluation.feedback import task_feedback
                        from sage_rl.evaluation.semantics import annotate_layout
                        semantic_layout = await asyncio.to_thread(annotate_layout, current_task, json.loads(scene_file.read_text()))
                        semantic_layout['_mesh_root'] = str(scene_file.parent)
                        feedback = task_feedback(current_task, semantic_layout)
                        payload["task_feedback"] = feedback
                        self._sage_feedback.append({"tool": tool_name, **feedback})
                        item.text = json.dumps(payload)
        return result

    self.call_tool_on_server = call_tool_with_layout_guard
    try:
        # Wrap process_query with BOTH:
        # 1. A watchdog task that periodically checks MCP subprocess liveness
        # 2. asyncio.wait_for as the hard wall-clock timeout
        process_task = asyncio.ensure_future(
            _ORIGINAL_PROCESS_QUERY(
                self,
                query,
                image_paths=image_paths,
                system_message=system_message,
            )
        )

        async def _mcp_liveness_watchdog():
            """Check every mcp_check_interval if any MCP subprocess died."""
            while not process_task.done():
                await asyncio.sleep(mcp_check_interval)
                if process_task.done():
                    break
                # Check all servers' subprocess PIDs
                for sname, sinfo in self.servers.items():
                    session = sinfo.get("session")
                    if session is None:
                        continue
                    # MCP stdio session has a _process or underlying subprocess
                    proc = getattr(session, "_process", None)
                    if proc is None:
                        # Try to find the subprocess via the transport
                        transport = getattr(session, "_transport", None)
                        if transport is not None:
                            proc = getattr(transport, "process", None)
                            if proc is None:
                                # anyio subprocess wrapper
                                proc = getattr(transport, "_proc", None)
                    if proc is not None:
                        # Check if the process is still alive
                        pid = getattr(proc, "pid", None) or getattr(proc, "_pid", None)
                        if pid is not None:
                            try:
                                os.kill(pid, 0)  # signal 0 = check if alive
                            except ProcessLookupError:
                                # MCP subprocess died (SIGSEGV/OOM/killed)
                                print(
                                    f"[SageRoomAgent] MCP server {sname} (pid={pid}) "
                                    f"died unexpectedly; aborting rollout "
                                    f"(layout_id={self.layout_id})",
                                    flush=True,
                                )
                                process_task.cancel()
                                return
                            except PermissionError:
                                pass  # process exists but not ours

        watchdog_task = asyncio.ensure_future(_mcp_liveness_watchdog())

        try:
            result = await asyncio.wait_for(
                asyncio.shield(process_task),
                timeout=session_timeout,
            )
            return result
        except asyncio.TimeoutError:
            self._sage_process_failure_reason = "process_query timeout"
            print(
                f"[SageRoomAgent] process_query timed out after "
                f"{session_timeout:.0f}s; aborting rollout "
                f"(layout_id={self.layout_id})",
                flush=True,
            )
            await _cleanup_mcp_client(self)
            return ("", False)
        except asyncio.CancelledError:
            # Watchdog cancelled the task (MCP subprocess died)
            self._sage_process_failure_reason = "MCP subprocess died"
            print(
                f"[SageRoomAgent] MCP subprocess died; cleanup + return failed "
                f"(layout_id={self.layout_id})",
                flush=True,
            )
            await _cleanup_mcp_client(self)
            return ("", False)
        finally:
            watchdog_task.cancel()
            try:
                await watchdog_task
            except (asyncio.CancelledError, Exception):
                pass
    finally:
        self.openai_client.chat.completions.create = original_create
        self.call_tool_on_server = original_call_tool


class SageRoomAgent:
    """Run the existing MCP room-generation loop with AReaL-provided OpenAI endpoint."""

    def __init__(
        self,
        server_paths: Optional[List[str]] = None,
        max_tool_calls: int = 15,
        mcp_mode: Optional[str] = None,
        use_final_appearance_critic: bool = True,
        sampling_temperature: float = TEMPERATURE,
    ):
        raw_paths = server_paths or [DEFAULT_SERVER_PATH]
        self.server_paths = [normalize_server_path(str(p)) for p in raw_paths]
        self.max_tool_calls = max_tool_calls
        self.mcp_mode = mcp_mode
        self.use_final_appearance_critic = use_final_appearance_critic
        self.sampling_temperature = sampling_temperature
        self.last_rollout_meta: Dict[str, Any] = {}

    def _resolve_server_paths(self, data: Dict[str, Any]) -> List[str]:
        if data.get("server_paths"):
            return [normalize_server_path(str(p)) for p in data["server_paths"]]
        return self.server_paths

    def _write_runtime_key_file(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
    ) -> str:
        payload = {
            "API_TOKEN": api_key,
            "API_TOKEN_QWEN": api_key,
            "API_URL_QWEN": base_url,
            "MODEL_NAME": model_name,
            "MODEL_DICT": {
                "qwen": model_name,
                "qwen_a": model_name,
                "openai": model_name,
            },
        }
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        )
        json.dump(payload, handle)
        handle.flush()
        handle.close()
        return handle.name

    def _prepare_task_payload(self, data: Dict[str, Any]) -> Dict[str, str]:
        room_desc = str(data["room_desc"]).strip()
        system_message = get_task_definition_text(room_desc, asset_mode="generation")
        max_calls = int(os.environ.get("SAGE_RL_MAX_TOOL_CALLS", str(self.max_tool_calls)))
        system_message += (
            f"\nTool budget: at most {max_calls} calls for the entire room. "
            "Make one tool call per assistant turn and inspect its result before the next action. "
            "Generate the room layout only once; correct the existing room afterward. "
            "Use the tool schema exactly, with JSON arguments and real object/room IDs from tool results."
        )
        user_message = f"Generate the room described below using generated assets:\n{room_desc}"
        return {
            "room_desc": room_desc,
            "system_message": system_message,
            "user_message": user_message,
        }

    async def run(self, data: dict, **extra_kwargs) -> float | Dict[str, float]:
        base_url = extra_kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY", "empty")
        model_name = data.get("model_name") or os.getenv(
            "SAGE_RL_MODEL_NAME",
            DEFAULT_RL_MODEL_PATH,
        )

        if not base_url:
            raise ValueError("base_url is required for SageRoomAgent RL rollout")

        self.last_rollout_meta = {}
        task_data = json.loads(data["task_json"]) if "task_json" in data else data
        validate_task(task_data)
        task = self._prepare_task_payload(data)
        client_key_file = self._write_runtime_key_file(base_url, api_key, model_name)
        server_key_file = os.environ.get(
            "SAGE_MCP_SERVER_KEY_FILE",
            str(DEFAULT_SERVER_KEY_FILE.resolve()),
        )

        prev_client_key = os.environ.get("SAGE_CLIENT_KEY_FILE")
        prev_server_key = os.environ.get("SAGE_KEY_FILE")
        prev_physics = os.environ.get("PHYSICS_CRITIC_ENABLED")
        prev_semantic = os.environ.get("SEMANTIC_CRITIC_ENABLED")
        os.environ["SAGE_CLIENT_KEY_FILE"] = client_key_file
        os.environ["SAGE_KEY_FILE"] = server_key_file
        os.environ.setdefault("PHYSICS_CRITIC_ENABLED", "false")
        os.environ.setdefault("SEMANTIC_CRITIC_ENABLED", "true")
        if self.mcp_mode:
            os.environ["SAGE_MCP_MODE"] = self.mcp_mode

        client = MCPClientOAI()
        client._sage_task = task_data
        client._sage_feedback = []
        # Bind the live Efficiency client to AReaL's per-episode policy proxy.
        import openai
        client.openai_client.close()
        external_config = os.environ.get("SAGE_EXTERNAL_ACTOR_CONFIG")
        client.external_actor = bool(external_config)
        if external_config:
            from sage_rl.agent.external_actor import ExternalActor
            client.openai_client = ExternalActor(
                external_config, model_name, os.environ.get("SAGE_EXTERNAL_ACTOR_AUDIT"))
        else:
            client.openai_client = openai.OpenAI(base_url=base_url, api_key=api_key)
        client.MODEL_NAME = model_name
        client.sampling_temperature = self.sampling_temperature
        client.sampling_seed = data.get("seed")
        client.process_query = _rl_process_query.__get__(client, MCPClientOAI)
        client.max_tool_calls = int(
            os.environ.get("SAGE_RL_MAX_TOOL_CALLS", str(self.max_tool_calls))
        )
        if self.mcp_mode:
            client.deployment_config["MCP_MODE"] = self.mcp_mode

        try:
            await client.connect_to_servers(self._resolve_server_paths(data))
            await client.process_query(
                task["user_message"],
                image_paths=None,
                system_message=task["system_message"],
            )
            layout_data = load_layout_data(client.layout_id)
            object_count = layout_object_count(layout_data)
            service_failure = str(
                getattr(client, "_sage_environment_failure_reason", "") or ""
            ) or rollout_infrastructure_failure(client.messages)
            infrastructure_failure = str(
                getattr(client, "_sage_process_failure_reason", "") or ""
            ) or service_failure
            appearance_result = None
            use_final_critic = self.use_final_appearance_critic
            env_flag = os.environ.get("SAGE_RL_FINAL_APPEARANCE_CRITIC")
            if env_flag is not None:
                use_final_critic = env_flag.lower() in {"1", "true", "yes", "on"}
            if use_final_critic and client.layout_id and not infrastructure_failure:
                from sage_rl.final_critic import (  # noqa: WPS433
                    run_final_appearance_critic,
                    save_final_critic_log,
                )

                os.environ.setdefault("SAGE_EVAL_ITERATIONS", "1")
                appearance_result = await run_final_appearance_critic(
                    layout_id=client.layout_id,
                    user_demand=task["room_desc"],
                    task_data=task_data,
                )
                save_final_critic_log(appearance_result)
                if appearance_result.get("environment_error"):
                    service_failure = appearance_result["environment_error"]
                    infrastructure_failure = service_failure
                elif not appearance_result.get("success"):
                    # The critic worker reports transport failures in its
                    # textual error field. Preserve that provenance so the
                    # bounded collection retry can recover after Isaac restarts.
                    error_text = str(appearance_result.get("error", "")).lower()
                    marker = next(
                        (m for m in _ISAAC_INFRASTRUCTURE_ERROR_MARKERS if m in error_text),
                        "",
                    )
                    if marker:
                        service_failure = marker
                        infrastructure_failure = marker
                if not appearance_result.get("success"):
                    print(
                        f"[SageRoomAgent] final critic failed: "
                        f"{appearance_result.get('error')}",
                        flush=True,
                    )

            # The final critic can update poses after physical settling.
            layout_data = load_layout_data(client.layout_id)
            from sage_rl.evaluation.semantics import annotate_layout
            if layout_data is not None:
                layout_data = await asyncio.to_thread(annotate_layout, task_data, layout_data, group_extras=True)
            reward, details = compute_episode_reward(
                layout_id=client.layout_id,
                appearance_result=appearance_result,
                layout_data=layout_data,
                task_data=task_data,
                tool_call_count=client.tool_call_count,
                tool_call_limit_reached=client.tool_call_limit_reached,
            )
            terminal_reward = float(reward)
            empty_layout = object_count == 0
            completion_ids = list(getattr(client, "_sage_completion_ids", []) or [])
            tool_names = tool_names_from_messages(client.messages)
            completion_rewards = {}
            if infrastructure_failure:
                completion_rewards = {
                    completion_id: 0.0 for completion_id in completion_ids
                }
            outcome = classify_outcome(
                strict_success=bool(details.get("strict_success", False)),
                process_error=str(getattr(client, "_sage_process_failure_reason", "") or ""),
                environment_error=service_failure,
                evaluation_error=("final verification unavailable" if
                    not details.get("evaluable") else ""),
                layout_exists=bool(client.layout_id),
            )
            reward_value = 0.0 if infrastructure_failure else terminal_reward
            layout_id = client.layout_id or ""
            self.last_rollout_meta = {
                "outcome": outcome.to_dict(),
                "sampling_contract": ({"external_actor": True, "model": model_name,
                    "max_output_tokens_including_reasoning": client.openai_client.output_tokens,
                    "context_truncation": "provider_limit", "seed_supported": False,
                    "wire_api": client.openai_client.wire_api,
                    "temperature": None if client.openai_client.wire_api == "responses" else self.sampling_temperature,
                    "thinking": "provider_default"}
                    if client.external_actor else {**sampling_contract(self.sampling_temperature),
                        "context_compaction": "overflow_only_completed_tool_exchanges_v1"}),
                "verification": details,
                "task_feedback": client._sage_feedback,
                "agent_reward": reward_value,
                "terminal_reward": terminal_reward,
                "layout_id": layout_id,
                "tool_call_count": int(client.tool_call_count),
                "max_tool_calls": int(client.max_tool_calls),
                "tool_call_limit_reached": client.tool_call_limit_reached,
                "completion_count": len(completion_ids),
                "context_budget_exhausted": bool(getattr(client, "context_budget_exhausted", False)),
                "context_compactions": getattr(client, "_sage_context_compactions", []),
                "tool_names": tool_names[: len(completion_ids)],
                "object_count": object_count,
                "empty_layout": int(empty_layout),
                "infrastructure_failure": int(bool(infrastructure_failure)),
                "failure_reason": infrastructure_failure,
                "blocked_layout_regenerations": int(
                    getattr(client, "_sage_blocked_layout_regenerations", 0) or 0
                ),
            }
            if "task_id" in data:
                self.last_rollout_meta["task_id"] = data["task_id"]
                self.last_rollout_meta["seed"] = data.get("seed")
            print(
                f"[SageRoomAgent] layout_id={layout_id} "
                f"reward={reward_value:.4f} details={details}",
                flush=True,
            )
            return completion_rewards or terminal_reward
        except Exception as exc:
            self.last_rollout_meta = {
                "agent_reward": 0.0, "terminal_reward": 0.0, "layout_id": "",
                "completion_count": len(getattr(client, "_sage_completion_ids", [])),
                "outcome": classify_outcome(strict_success=False, process_error=repr(exc)).to_dict(),
            }
            print(
                f"[SageRoomAgent] rollout failed, reward=0.0 error={exc!r}",
                flush=True,
            )
            return 0.0
        finally:
            await _cleanup_mcp_client(client)
            if prev_client_key is None:
                os.environ.pop("SAGE_CLIENT_KEY_FILE", None)
            else:
                os.environ["SAGE_CLIENT_KEY_FILE"] = prev_client_key
            if prev_server_key is None:
                os.environ.pop("SAGE_KEY_FILE", None)
            else:
                os.environ["SAGE_KEY_FILE"] = prev_server_key
            if prev_physics is None:
                os.environ.pop("PHYSICS_CRITIC_ENABLED", None)
            else:
                os.environ["PHYSICS_CRITIC_ENABLED"] = prev_physics
            if prev_semantic is None:
                os.environ.pop("SEMANTIC_CRITIC_ENABLED", None)
            else:
                os.environ["SEMANTIC_CRITIC_ENABLED"] = prev_semantic
            try:
                os.unlink(client_key_file)
            except OSError:
                pass






