from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import pytest

from mcp_tool_harness.agent.deepseek import DeepSeekConfig, DeepSeekToolAgent
from mcp_tool_harness.server import ToolGateway


class FakeDeepSeekClient:
    def __init__(self, replies: Sequence[Mapping[str, Any]]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        self.requests.append(
            {
                "messages": [dict(message) for message in messages],
                "tools": [dict(tool) for tool in tools or ()],
                "extra_body": dict(extra_body or {}),
            }
        )
        return self.replies.pop(0)


@pytest.mark.asyncio
async def test_deepseek_agent_invokes_gateway_tool_and_returns_final_answer() -> None:
    gateway = ToolGateway(default_rate_limit_per_minute=None)
    gateway.register_tool(
        "math.add",
        lambda left, right: {"value": left + right},
        description="Add two integers.",
        input_schema={
            "type": "object",
            "properties": {
                "left": {"type": "integer"},
                "right": {"type": "integer"},
            },
            "required": ["left", "right"],
        },
    )
    client = FakeDeepSeekClient(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tool-call-1",
                        "type": "function",
                        "function": {
                            "name": "math_add",
                            "arguments": '{"left": 2, "right": 3}',
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "2 + 3 = 5"},
        ]
    )
    agent = DeepSeekToolAgent(gateway, client)

    result = await agent.arun("计算 2 + 3", request_id="req-1")

    assert result.content == "2 + 3 = 5"
    assert result.tool_invocations[0].tool_name == "math.add"
    assert result.tool_invocations[0].result == {"value": 5}
    assert client.requests[0]["tools"][0]["function"]["name"] == "math_add"
    tool_message = client.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert json.loads(tool_message["content"]) == {
        "ok": True,
        "tool_name": "math.add",
        "result": {"value": 5},
        "cached": False,
    }


@pytest.mark.asyncio
async def test_deepseek_agent_sends_tool_errors_back_to_model() -> None:
    gateway = ToolGateway(default_rate_limit_per_minute=None)
    gateway.register_tool(
        "math.add",
        lambda left, right: {"value": left + right},
        input_schema={
            "type": "object",
            "properties": {
                "left": {"type": "integer"},
                "right": {"type": "integer"},
            },
            "required": ["left", "right"],
        },
    )
    client = FakeDeepSeekClient(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tool-call-1",
                        "type": "function",
                        "function": {
                            "name": "math_add",
                            "arguments": '{"left": 2}',
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "工具参数缺少 right。"},
        ]
    )
    agent = DeepSeekToolAgent(gateway, client)

    result = await agent.arun("计算 2 + 3")

    assert result.content == "工具参数缺少 right。"
    assert result.tool_invocations[0].ok is False
    payload = json.loads(client.requests[1]["messages"][-1]["content"])
    assert payload["ok"] is False
    assert payload["error_type"] == "ToolInputValidationError"
    assert payload["error_code"] == "tool_input_validation_error"


def test_deepseek_config_accepts_base_or_full_completion_url() -> None:
    assert DeepSeekConfig(api_key="key", base_url="https://api.deepseek.com").completion_url == (
        "https://api.deepseek.com/chat/completions"
    )
    assert DeepSeekConfig(api_key="key", base_url="https://llm.example.com/v1/chat/completions").completion_url == (
        "https://llm.example.com/v1/chat/completions"
    )


@pytest.mark.asyncio
async def test_deepseek_agent_preserves_reasoning_content_for_tool_followup() -> None:
    gateway = ToolGateway(default_rate_limit_per_minute=None)
    gateway.register_tool("text.echo", lambda text: {"text": text})
    client = FakeDeepSeekClient(
        [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "I should call the echo tool.",
                "tool_calls": [
                    {
                        "id": "tool-call-1",
                        "type": "function",
                        "function": {
                            "name": "text_echo",
                            "arguments": '{"text": "hello"}',
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "hello"},
        ]
    )
    agent = DeepSeekToolAgent(gateway, client)

    await agent.arun("echo hello")

    assistant_message = client.requests[1]["messages"][-2]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["reasoning_content"] == "I should call the echo tool."


def test_deepseek_config_rejects_invalid_runtime_limits() -> None:
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        DeepSeekConfig(api_key="key", timeout_seconds=0)
    with pytest.raises(ValueError, match="max_tool_rounds must be positive"):
        DeepSeekConfig(api_key="key", max_tool_rounds=0)
