"""Minimal agents built on top of MCP Tool Harness gateways."""

from importlib import import_module
from typing import Any

__all__ = [
    "AgentResult",
    "DeepSeekAPIError",
    "DeepSeekChatClient",
    "DeepSeekConfig",
    "DeepSeekToolAgent",
    "ToolInvocation",
    "create_deepseek_agent",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(".deepseek", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
