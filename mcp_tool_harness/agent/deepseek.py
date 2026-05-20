"""A small DeepSeek-backed agent for exercising registered harness tools.

The implementation targets DeepSeek/OpenAI-compatible Chat Completions with
function calling. It intentionally uses only Python's standard library so the
core package stays dependency-free.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Mapping, Sequence
from uuid import uuid4

from mcp_tool_harness.server import ToolGateway, ToolHarnessError


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_SYSTEM_PROMPT = (
    "You are a focused tool-testing agent. Use the provided tools when they are "
    "needed, pass arguments exactly as JSON objects, and summarize tool results "
    "clearly for the user."
)
_FUNCTION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class DeepSeekAgentError(RuntimeError):
    """Base error for DeepSeek agent failures."""


class DeepSeekAPIError(DeepSeekAgentError):
    """Raised when the DeepSeek-compatible API call fails."""


@dataclass(frozen=True)
class DeepSeekConfig:
    """Connection settings for a DeepSeek/OpenAI-compatible endpoint."""

    api_key: str
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    model: str = DEFAULT_DEEPSEEK_MODEL
    timeout_seconds: float = 30.0
    temperature: float = 0.0
    max_tool_rounds: int = 4

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be positive")

    @classmethod
    def from_env(
        cls,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        temperature: float | None = None,
        max_tool_rounds: int | None = None,
    ) -> "DeepSeekConfig":
        return cls(
            api_key=api_key if api_key is not None else os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=base_url or os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
            model=model or os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
            timeout_seconds=timeout_seconds
            if timeout_seconds is not None
            else float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "30")),
            temperature=temperature if temperature is not None else float(os.getenv("DEEPSEEK_TEMPERATURE", "0")),
            max_tool_rounds=max_tool_rounds
            if max_tool_rounds is not None
            else int(os.getenv("DEEPSEEK_MAX_TOOL_ROUNDS", "4")),
        )

    @property
    def completion_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"


class DeepSeekChatClient:
    """Thin HTTP client for DeepSeek-compatible chat completions."""

    def __init__(self, config: DeepSeekConfig) -> None:
        self.config = config

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return await asyncio.to_thread(
            self.complete_sync,
            messages,
            tools=tools,
            extra_body=extra_body,
        )

    def complete_sync(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        if not self.config.api_key.strip():
            raise DeepSeekAPIError("DEEPSEEK_API_KEY is required")

        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [dict(message) for message in messages],
            "temperature": self.config.temperature,
        }
        if tools:
            body["tools"] = [dict(tool) for tool in tools]
            body["tool_choice"] = "auto"
        if extra_body:
            body.update(dict(extra_body))

        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.config.completion_url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            # API key 只放在请求头，不写日志、不写异常消息，避免凭证泄露。
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise DeepSeekAPIError(f"DeepSeek API HTTP {exc.code}: {_compact_error(detail)}") from exc
        except urllib.error.URLError as exc:
            raise DeepSeekAPIError(f"DeepSeek API request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise DeepSeekAPIError(f"DeepSeek API timed out after {self.config.timeout_seconds}s") from exc

        try:
            decoded = json.loads(response_body)
            return decoded["choices"][0]["message"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise DeepSeekAPIError("DeepSeek API response does not contain choices[0].message") from exc


@dataclass(frozen=True)
class ToolInvocation:
    """One tool call made by the model and executed through ToolGateway."""

    tool_call_id: str
    model_tool_name: str
    tool_name: str
    arguments: Mapping[str, Any]
    ok: bool
    result: Any = None
    error: str | None = None
    error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "model_tool_name": self.model_tool_name,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "ok": self.ok,
            "result": self.result,
            "error": self.error,
            "error_type": self.error_type,
        }


@dataclass(frozen=True)
class AgentResult:
    """Final agent answer plus the full trace useful for testing tools."""

    content: str
    messages: Sequence[Mapping[str, Any]]
    tool_invocations: Sequence[ToolInvocation] = field(default_factory=tuple)
    raw_message: Mapping[str, Any] = field(default_factory=dict)


class DeepSeekToolAgent:
    """Minimal tool-calling agent backed by a ToolGateway.

    Thread-safety说明：
    - Agent 不保存单次运行状态；messages 与 tool_invocations 都在 run/arun 调用栈内创建。
    - ToolGateway 内部的限流、熔断、幂等仍然是工具执行的并发控制边界。

    事务边界说明：
    - Agent 只编排模型和工具调用，不开启数据库事务。
    - 写工具必须由 handler 或下游服务自己保证幂等、回滚和补偿。
    """

    def __init__(
        self,
        gateway: ToolGateway,
        client: Any,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        principal: str = "deepseek-agent",
        max_tool_rounds: int | None = None,
    ) -> None:
        self.gateway = gateway
        self.client = client
        self.system_prompt = system_prompt
        self.principal = principal
        self.max_tool_rounds = max_tool_rounds

    def run(
        self,
        prompt: str,
        *,
        history: Sequence[Mapping[str, Any]] | None = None,
        principal: str | None = None,
        request_id: str | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> AgentResult:
        return asyncio.run(
            self.arun(
                prompt,
                history=history,
                principal=principal,
                request_id=request_id,
                extra_body=extra_body,
            )
        )

    async def arun(
        self,
        prompt: str,
        *,
        history: Sequence[Mapping[str, Any]] | None = None,
        principal: str | None = None,
        request_id: str | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> AgentResult:
        if not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        call_request_id = request_id or f"agent-{uuid4().hex}"
        caller = principal or self.principal
        tools, alias_to_tool = _gateway_tools_for_model(self.gateway)
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(dict(item) for item in history or ())
        messages.append({"role": "user", "content": prompt})
        invocations: list[ToolInvocation] = []

        for _round_index in range(self._resolve_max_tool_rounds()):
            assistant_message = dict(
                await self.client.complete(
                    messages,
                    tools=tools,
                    extra_body=extra_body,
                )
            )
            tool_calls = _extract_tool_calls(assistant_message)
            messages.append(_assistant_history_message(assistant_message))
            if not tool_calls:
                return AgentResult(
                    content=str(assistant_message.get("content") or ""),
                    messages=tuple(messages),
                    tool_invocations=tuple(invocations),
                    raw_message=assistant_message,
                )

            for tool_call in tool_calls:
                invocation, tool_message = await self._invoke_tool_call(
                    tool_call,
                    alias_to_tool=alias_to_tool,
                    principal=caller,
                    request_id=call_request_id,
                )
                invocations.append(invocation)
                messages.append(tool_message)

        content = "工具调用轮次已达到上限，已停止继续调用。请检查工具是否反复返回无法收敛的结果。"
        return AgentResult(
            content=content,
            messages=tuple(messages),
            tool_invocations=tuple(invocations),
            raw_message={"role": "assistant", "content": content},
        )

    def _resolve_max_tool_rounds(self) -> int:
        if self.max_tool_rounds is not None:
            return self.max_tool_rounds
        config = getattr(self.client, "config", None)
        return int(getattr(config, "max_tool_rounds", 4))

    async def _invoke_tool_call(
        self,
        tool_call: Mapping[str, Any],
        *,
        alias_to_tool: Mapping[str, str],
        principal: str,
        request_id: str,
    ) -> tuple[ToolInvocation, dict[str, Any]]:
        tool_call_id = str(tool_call.get("id") or f"call_{uuid4().hex}")
        function = tool_call.get("function") or {}
        model_tool_name = str(function.get("name") or "")
        tool_name = alias_to_tool.get(model_tool_name)

        try:
            if tool_name is None:
                raise ValueError(f"model requested unknown tool '{model_tool_name}'")
            arguments = _parse_tool_arguments(function.get("arguments"))
            # 工具执行统一回到 ToolGateway，限流、超时、熔断、幂等不在 agent 内重复实现。
            response = await self.gateway.ainvoke(
                tool_name,
                arguments,
                principal=principal,
                request_id=f"{request_id}:{tool_call_id}",
            )
            payload = {
                "ok": True,
                "tool_name": tool_name,
                "result": response.result,
                "cached": response.cached,
            }
            invocation = ToolInvocation(
                tool_call_id=tool_call_id,
                model_tool_name=model_tool_name,
                tool_name=tool_name,
                arguments=arguments,
                ok=True,
                result=response.result,
            )
        except Exception as exc:  # noqa: BLE001 - tool failures must be sent back to the model.
            arguments = _safe_parse_tool_arguments(function.get("arguments"))
            payload = _tool_error_payload(tool_name or model_tool_name, exc)
            invocation = ToolInvocation(
                tool_call_id=tool_call_id,
                model_tool_name=model_tool_name,
                tool_name=tool_name or model_tool_name,
                arguments=arguments,
                ok=False,
                error=payload["error"],
                error_type=payload["error_type"],
            )

        return invocation, {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": model_tool_name,
            "content": json.dumps(payload, ensure_ascii=False, default=str),
        }


def create_deepseek_agent(
    gateway: ToolGateway,
    *,
    config: DeepSeekConfig | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    principal: str = "deepseek-agent",
) -> DeepSeekToolAgent:
    resolved_config = config or DeepSeekConfig.from_env(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    return DeepSeekToolAgent(
        gateway,
        DeepSeekChatClient(resolved_config),
        system_prompt=system_prompt,
        principal=principal,
    )


def _gateway_tools_for_model(gateway: ToolGateway) -> tuple[list[dict[str, Any]], dict[str, str]]:
    aliases: dict[str, str] = {}
    tools: list[dict[str, Any]] = []
    for metadata in gateway.list_tools():
        original_name = str(metadata["name"])
        alias = _tool_alias(original_name, aliases)
        aliases[alias] = original_name
        description = str(metadata.get("description") or f"Call tool {original_name}.")
        if alias != original_name:
            description = f"{description} Original harness tool name: {original_name}."
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": alias,
                    "description": description,
                    "parameters": _normalize_parameters(metadata.get("input_schema")),
                },
            }
        )
    return tools, aliases


def _tool_alias(name: str, existing: Mapping[str, str]) -> str:
    candidate = re.sub(r"[^A-Za-z0-9_-]", "_", name).strip("_")
    if not candidate:
        candidate = f"tool_{sha256(name.encode('utf-8')).hexdigest()[:16]}"
    candidate = candidate[:64]
    if _FUNCTION_NAME_PATTERN.fullmatch(candidate) and candidate not in existing:
        return candidate

    prefix = candidate[:56] or "tool"
    index = 1
    while True:
        alias = f"{prefix}_{index}"
        if alias not in existing:
            return alias[:64]
        index += 1


def _normalize_parameters(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, Mapping) or not schema:
        return {"type": "object", "properties": {}}
    parameters = dict(schema)
    if parameters.get("type") not in (None, "object"):
        return {"type": "object", "properties": {}}
    parameters.setdefault("type", "object")
    parameters.setdefault("properties", {})
    return parameters


def _extract_tool_calls(message: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        return []
    return [call for call in calls if isinstance(call, Mapping)]


def _assistant_history_message(message: Mapping[str, Any]) -> dict[str, Any]:
    history: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
    # DeepSeek thinking mode requires reasoning_content to be passed back after
    # any assistant turn that performs tool calls; dropping it causes HTTP 400.
    if "reasoning_content" in message:
        history["reasoning_content"] = message.get("reasoning_content")
    tool_calls = _extract_tool_calls(message)
    if tool_calls:
        history["tool_calls"] = [dict(call) for call in tool_calls]
    return history


def _parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    if raw_arguments in (None, ""):
        return {}
    if isinstance(raw_arguments, Mapping):
        return dict(raw_arguments)
    if not isinstance(raw_arguments, str):
        raise ValueError("tool arguments must be a JSON object")
    decoded = json.loads(raw_arguments)
    if not isinstance(decoded, Mapping):
        raise ValueError("tool arguments must be a JSON object")
    return dict(decoded)


def _safe_parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    try:
        return _parse_tool_arguments(raw_arguments)
    except Exception:  # noqa: BLE001 - this is best-effort trace data.
        return {}


def _tool_error_payload(tool_name: str, exc: Exception) -> dict[str, Any]:
    payload = {
        "ok": False,
        "tool_name": tool_name,
        "error_type": exc.__class__.__name__,
        "error": str(exc),
    }
    if isinstance(exc, ToolHarnessError):
        payload["error_code"] = exc.error_code
    return payload


def _compact_error(body: str) -> str:
    text = body.strip()
    if not text:
        return "<empty response body>"
    if len(text) > 500:
        return f"{text[:500]}..."
    return text


def _build_demo_gateway() -> ToolGateway:
    gateway = ToolGateway(default_rate_limit_per_minute=120, default_timeout_ms=2_000)

    def add(left: int, right: int) -> dict[str, int]:
        return {"value": left + right}

    def echo(text: str) -> dict[str, str]:
        return {"text": text}

    gateway.register_tool(
        "math.add",
        add,
        description="Add two integers and return their sum.",
        input_schema={
            "type": "object",
            "properties": {
                "left": {"type": "integer"},
                "right": {"type": "integer"},
            },
            "required": ["left", "right"],
        },
        timeout_ms=500,
    )
    gateway.register_tool(
        "text.echo",
        echo,
        description="Echo a text string.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        timeout_ms=500,
    )
    return gateway


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a small DeepSeek agent against demo harness tools.")
    parser.add_argument("prompt", nargs="+", help="Prompt to send to the agent.")
    parser.add_argument("--api-key", default=None, help="DeepSeek API key. Defaults to DEEPSEEK_API_KEY.")
    parser.add_argument("--base-url", default=None, help="DeepSeek-compatible base URL.")
    parser.add_argument("--model", default=None, help="DeepSeek model name.")
    parser.add_argument("--show-tool-results", action="store_true", help="Print tool invocation trace after the answer.")
    args = parser.parse_args(argv)

    agent = create_deepseek_agent(
        _build_demo_gateway(),
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )
    result = agent.run(" ".join(args.prompt))
    print(result.content)
    if args.show_tool_results:
        print(json.dumps([item.to_dict() for item in result.tool_invocations], ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised manually.
    raise SystemExit(main())
