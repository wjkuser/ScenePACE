"""Opt-in external actor transport; never applies Qwen-specific token controls."""

import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

from openai import OpenAI
from openai.types.chat import ChatCompletion


class ExternalActor:
    def __init__(self, config_file, model, audit_file=None):
        cfg = json.loads(Path(config_file).read_text())
        self.model = model
        self.output_tokens = int(cfg.get("output_tokens", 16384))
        self.reasoning_effort = cfg.get("reasoning_effort", "medium")
        self.wire_api = cfg.get("wire_api", "chat")
        self.client = OpenAI(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            default_headers=cfg.get("headers", {}),
            timeout=300,
            max_retries=2,
        )
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.reasoning = {}
        self.response_items = {}
        self.audit_file = audit_file

    def close(self):
        self.client.close()

    def create(self, **kwargs):
        messages = copy.deepcopy(kwargs["messages"])
        # Native client stores display fields that are not part of either API.
        allowed = {
            "role",
            "content",
            "tool_calls",
            "tool_call_id",
            "name",
            "reasoning_content",
        }
        messages = [{k: v for k, v in m.items() if k in allowed} for m in messages]
        tools = kwargs.get("tools", [])
        started = time.monotonic()
        if self.wire_api == "responses":
            inputs = []
            for msg in messages:
                if msg["role"] == "tool":
                    inputs.append(
                        dict(
                            type="function_call_output",
                            call_id=msg["tool_call_id"],
                            output=msg["content"],
                        )
                    )
                elif msg.get("tool_calls"):
                    calls = msg["tool_calls"]
                    cached = self.response_items.get(calls[0]["id"])
                    if cached:
                        # Preserve opaque reasoning items across tool iterations.
                        inputs.extend(cached)
                    else:
                        if msg.get("content"):
                            inputs.append(
                                dict(role="assistant", content=msg["content"])
                            )
                        inputs.extend(
                            dict(type="function_call", call_id=c["id"], **c["function"])
                            for c in calls
                        )
                else:
                    inputs.append(
                        dict(role=msg["role"], content=msg.get("content") or "")
                    )
            req = dict(
                model=self.model,
                input=inputs,
                max_output_tokens=self.output_tokens,
                tools=[dict(type="function", **t["function"]) for t in tools],
                parallel_tool_calls=False,
                store=False,
                reasoning={"effort": self.reasoning_effort},
                include=["reasoning.encrypted_content"],
            )
            raw = self.client.responses.create(**req)
            calls = [
                dict(
                    id=o.call_id,
                    type="function",
                    function=dict(name=o.name, arguments=o.arguments),
                )
                for o in raw.output
                if o.type == "function_call"
            ]
            if calls:
                self.response_items[calls[0]["id"]] = [
                    o.model_dump(exclude_none=True) for o in raw.output
                ]
            usage = raw.usage
            response = ChatCompletion(
                id=raw.id,
                object="chat.completion",
                created=int(time.time()),
                model=raw.model,
                choices=[
                    dict(
                        index=0,
                        finish_reason="length"
                        if raw.status == "incomplete"
                        else ("tool_calls" if calls else "stop"),
                        message=dict(
                            role="assistant",
                            content=raw.output_text or None,
                            tool_calls=calls or None,
                        ),
                    )
                ],
                usage=dict(
                    prompt_tokens=usage.input_tokens,
                    completion_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                ),
            )
        else:
            for msg in messages:
                if msg.get("tool_calls"):
                    reasoning = self.reasoning.get(msg["tool_calls"][0]["id"])
                    if reasoning is not None:
                        msg["reasoning_content"] = reasoning
            req = dict(
                model=self.model,
                messages=messages,
                max_tokens=self.output_tokens,
                tools=tools,
                temperature=kwargs.get("temperature", 0.6),
                top_p=0.95,
            )
            # Both providers accept single-call sampling; seeds are deliberately
            # omitted because reproducibility is not promised by these APIs.
            req["parallel_tool_calls"] = False
            raw = response = self.client.chat.completions.create(**req)
            msg = response.choices[0].message
            if msg.tool_calls:
                self.reasoning[msg.tool_calls[0].id] = getattr(
                    msg, "reasoning_content", None
                )
        if self.audit_file:
            with Path(self.audit_file).open("a") as f:
                f.write(
                    json.dumps(
                        dict(
                            model=raw.model,
                            wire_api=self.wire_api,
                            response_id=raw.id,
                            seconds=time.monotonic() - started,
                            usage=raw.usage.model_dump() if raw.usage else None,
                            output_budget=self.output_tokens,
                            seed_supported=False,
                            finish_reason=response.choices[0].finish_reason,
                        )
                    )
                    + "\n"
                )
        return response
