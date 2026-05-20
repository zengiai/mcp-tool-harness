"""YAML-backed dynamic Harness configuration.

This module keeps configuration loading outside Gateway/Security so a future
registry center can implement the same PolicyConfigSource contract and push
ToolPolicy plus MCP server connection metadata into Registry.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from mcp_tool_harness.core.models import AuthType, ToolPolicy, ToolServer, ToolSpec, stable_json_hash
from mcp_tool_harness.mcp import MCPClient, discover_tools


class ConfigLoadError(ValueError):
    """Raised when a policy configuration file cannot be parsed or validated."""


class PolicyConfigSource(Protocol):
    """Common contract for YAML files, Nacos, Apollo, etcd, or any config center."""

    async def load(self) -> "HarnessConfig":
        """Load the latest config snapshot."""


@dataclass(frozen=True)
class HarnessConfig:
    """Normalized configuration snapshot loaded from an external source."""

    policies: tuple[ToolPolicy, ...] = field(default_factory=tuple)
    mcp_servers: tuple["MCPServerConfig", ...] = field(default_factory=tuple)
    raw: Mapping[str, Any] = field(default_factory=dict)
    version: str = ""


@dataclass(frozen=True)
class MCPServerConfig:
    """Connection metadata for one upstream MCP server."""

    server_id: str
    transport: str = "streamable_http"
    name: str = ""
    version: str = "1.0.0"
    url: str | None = None
    sse_url: str | None = None
    message_endpoint: str | None = None
    command: tuple[str, ...] = field(default_factory=tuple)
    args: tuple[str, ...] = field(default_factory=tuple)
    cwd: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout_ms: int | None = None
    auto_initialize: bool = False
    auth_type: AuthType = AuthType.NONE
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.server_id or not isinstance(self.server_id, str):
            raise ConfigLoadError("mcp_servers[].server_id must be a non-empty string")
        transport = str(self.transport).strip().lower().replace("-", "_")
        object.__setattr__(self, "transport", transport)
        object.__setattr__(self, "name", self.name or self.server_id)
        object.__setattr__(self, "command", tuple(self.command))
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "env", dict(self.env))
        object.__setattr__(self, "headers", dict(self.headers))
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.timeout_ms is not None and self.timeout_ms <= 0:
            raise ConfigLoadError("mcp_servers[].timeout_ms must be positive")

    @property
    def timeout_seconds(self) -> float:
        return (self.timeout_ms or 30_000) / 1000

    @property
    def endpoint(self) -> str:
        if self.transport in {"streamable_http", "http", "https"}:
            return str(self.url or "")
        if self.transport == "sse":
            return str(self.sse_url or self.url or "")
        if self.transport == "stdio":
            return " ".join((*self.command, *self.args))
        return str(self.url or self.sse_url or "")

    @property
    def command_line(self) -> tuple[str, ...]:
        return (*self.command, *self.args)


@dataclass(frozen=True)
class MCPDiscoveryResult:
    """MCP discovery output for one configured server."""

    server: MCPServerConfig
    client: Any
    tools: tuple[ToolSpec, ...]


@dataclass(frozen=True)
class MCPBootstrapResult:
    """Result returned after registering configured MCP servers and tools."""

    router: "MCPClientRouter"
    results: tuple[MCPDiscoveryResult, ...]

    @property
    def tools(self) -> tuple[ToolSpec, ...]:
        return tuple(tool for result in self.results for tool in result.tools)


class MCPClientRouter:
    """Route core gateway calls to the MCP client that owns a tool server."""

    def __init__(self, clients: Mapping[str, Any]) -> None:
        if not clients:
            raise ConfigLoadError("at least one MCP client is required")
        self._clients = dict(clients)

    @property
    def clients(self) -> Mapping[str, Any]:
        return dict(self._clients)

    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        tool_spec: Any | None = None,
        context: Any | None = None,
    ) -> Any:
        del context
        server_id = getattr(tool_spec, "server_id", None)
        tool_name = name
        if server_id is None and "/" in name:
            server_id, tool_name = name.split("/", 1)
        if server_id is None:
            if len(self._clients) != 1:
                raise ConfigLoadError("server_id is required when multiple MCP clients are configured")
            server_id = next(iter(self._clients))
        client = self._clients.get(str(server_id))
        if client is None:
            raise ConfigLoadError(f"no MCP client configured for server_id {server_id!r}")
        return client.call_tool(tool_name, dict(arguments or {}))

    def list_tools(self, server_id: str | None = None) -> Any:
        if server_id is None:
            if len(self._clients) != 1:
                raise ConfigLoadError("server_id is required when multiple MCP clients are configured")
            server_id = next(iter(self._clients))
        client = self._clients.get(server_id)
        if client is None:
            raise ConfigLoadError(f"no MCP client configured for server_id {server_id!r}")
        return client.list_tools()

    def close(self) -> None:
        for client in self._clients.values():
            closer = getattr(client, "close", None)
            if closer is not None:
                closer()


@dataclass(slots=True)
class YamlConfigSource:
    """Load ToolPolicy configuration from a YAML file.

    线程安全说明：
    - 该类不缓存可变状态，每次 load 都从文件读取最新快照。
    - 后续接配置中心时保持 load() 返回 HarnessConfig 即可复用 apply_policy_config。
    """

    path: str | Path

    async def load(self) -> HarnessConfig:
        text = await asyncio.to_thread(Path(self.path).read_text, encoding="utf-8")
        return load_yaml_config(text)

    async def apply_to(self, registry: Any) -> tuple[ToolPolicy, ...]:
        """Load and apply the latest YAML snapshot to a Registry-like object."""

        return await apply_policy_config(registry, await self.load())

    async def discover_mcp_to(
        self,
        registry: Any,
        *,
        client_factory: Any | None = None,
    ) -> MCPBootstrapResult:
        """Load YAML, discover configured MCP servers, and register their tools."""

        return await discover_and_register_mcp_servers(
            registry,
            await self.load(),
            client_factory=client_factory,
        )


def load_yaml_config(text: str) -> HarnessConfig:
    """Parse YAML text and return a normalized HarnessConfig.

    PyYAML is used when installed.  A small built-in YAML subset parser is kept
    for the SDK test/runtime path so core policy loading has no hard dependency.
    """

    data = _load_yaml_mapping(text)
    root = _select_root(data)
    policies = tuple(_tool_policy_from_mapping(item, index) for index, item in enumerate(_policy_items(root)))
    mcp_servers = tuple(
        _mcp_server_from_mapping(item, index)
        for index, item in enumerate(_mcp_server_items(root))
    )
    return HarnessConfig(
        policies=policies,
        mcp_servers=mcp_servers,
        raw=data,
        version=str(root.get("version") or stable_json_hash(data)[:16]),
    )


async def apply_policy_config(registry: Any, config: HarnessConfig) -> tuple[ToolPolicy, ...]:
    """Apply a normalized config snapshot to a Registry-like object.

    Existing policies with the same deterministic policy_id are overwritten by
    Registry.register_policy, so config updates take effect for later calls.
    """

    registrar = getattr(registry, "register_policy", None)
    if registrar is None:
        raise ConfigLoadError("registry must expose register_policy(policy)")

    applied: list[ToolPolicy] = []
    for policy in config.policies:
        result = registrar(policy)
        if hasattr(result, "__await__"):
            result = await result
        applied.append(result)
    return tuple(applied)


async def discover_and_register_mcp_servers(
    registry: Any,
    config_or_servers: HarnessConfig | Sequence[MCPServerConfig],
    *,
    client_factory: Any | None = None,
) -> MCPBootstrapResult:
    """Create MCP clients, discover tools, and register servers/tools into Registry.

    The function only writes metadata.  Call execution still goes through
    core.ToolGateway using the returned MCPClientRouter.
    """

    servers = (
        config_or_servers.mcp_servers
        if isinstance(config_or_servers, HarnessConfig)
        else tuple(config_or_servers)
    )
    register_server = getattr(registry, "register_server", None)
    register_tool = getattr(registry, "register_tool", None)
    if register_server is None or register_tool is None:
        raise ConfigLoadError("registry must expose register_server(server) and register_tool(spec)")

    make_client = client_factory or create_mcp_client
    results: list[MCPDiscoveryResult] = []
    clients: dict[str, Any] = {}
    for server_config in servers:
        client = make_client(server_config)
        raw_tools = await asyncio.to_thread(discover_tools, client)
        discovered: list[ToolSpec] = []
        await _maybe_await(
            register_server(
                ToolServer(
                    server_id=server_config.server_id,
                    endpoint=server_config.endpoint,
                    name=server_config.name,
                    version=server_config.version,
                    auth_type=server_config.auth_type,
                    capabilities=server_config.capabilities,
                    metadata={
                        **dict(server_config.metadata),
                        "transport": server_config.transport,
                    },
                )
            )
        )
        for mcp_tool in raw_tools:
            spec = ToolSpec(
                name=mcp_tool.name,
                description=mcp_tool.description or f"MCP tool {server_config.server_id}/{mcp_tool.name}",
                input_schema=mcp_tool.input_schema,
                output_schema=mcp_tool.output_schema or {},
                server_id=server_config.server_id,
                version=server_config.version,
                metadata={
                    "annotations": dict(mcp_tool.annotations),
                    "mcp_raw": dict(mcp_tool.raw),
                    "mcp_server_id": server_config.server_id,
                },
            )
            registered = await _maybe_await(register_tool(spec))
            discovered.append(registered)
        clients[server_config.server_id] = client
        results.append(MCPDiscoveryResult(server=server_config, client=client, tools=tuple(discovered)))

    return MCPBootstrapResult(router=MCPClientRouter(clients), results=tuple(results))


def create_mcp_client(server: MCPServerConfig, *, environ: Mapping[str, str] | None = None) -> MCPClient:
    """Create an MCPClient from one MCPServerConfig."""

    env_source = os.environ if environ is None else environ
    headers = _expand_mapping(server.headers, env_source)
    env = _expand_mapping(server.env, env_source) if server.env else None
    if server.transport in {"streamable_http", "http", "https"}:
        if not server.url:
            raise ConfigLoadError(f"mcp server {server.server_id!r} requires url")
        return MCPClient.from_streamable_http(
            _expand_env(str(server.url), env_source),
            headers=headers,
            default_timeout=server.timeout_seconds,
            auto_initialize=server.auto_initialize,
        )
    if server.transport == "sse":
        endpoint = server.sse_url or server.url
        if not endpoint:
            raise ConfigLoadError(f"mcp server {server.server_id!r} requires sse_url or url")
        return MCPClient.from_sse(
            _expand_env(str(endpoint), env_source),
            message_endpoint=server.message_endpoint,
            headers=headers,
            default_timeout=server.timeout_seconds,
            auto_initialize=server.auto_initialize,
        )
    if server.transport == "stdio":
        command = tuple(_expand_env(item, env_source) for item in server.command_line)
        if not command:
            raise ConfigLoadError(f"mcp server {server.server_id!r} requires command")
        return MCPClient.from_stdio(
            command,
            cwd=_expand_env(server.cwd, env_source) if server.cwd else None,
            env=env,
            default_timeout=server.timeout_seconds,
            auto_initialize=server.auto_initialize,
        )
    raise ConfigLoadError(f"unsupported MCP transport: {server.transport}")


def _load_yaml_mapping(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        loaded = _parse_yaml_subset(text)
    else:
        loaded = yaml.safe_load(text)  # type: ignore[no-any-return]

    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ConfigLoadError("YAML root must be a mapping")
    return dict(loaded)


def _select_root(data: Mapping[str, Any]) -> Mapping[str, Any]:
    if "tool_harness" in data:
        root = data["tool_harness"]
    elif "mcp_tool_harness" in data:
        root = data["mcp_tool_harness"]
    else:
        root = data
    if not isinstance(root, Mapping):
        raise ConfigLoadError("tool_harness root must be a mapping")
    return root


def _policy_items(root: Mapping[str, Any]) -> Sequence[Any]:
    policies = root.get("policies", root.get("tool_policies", ()))
    if policies is None:
        return ()
    if not isinstance(policies, Sequence) or isinstance(policies, (str, bytes)):
        raise ConfigLoadError("policies must be a list")
    return policies


def _mcp_server_items(root: Mapping[str, Any]) -> Sequence[Any]:
    servers = root.get("mcp_servers", ())
    if servers is None:
        return ()
    if not isinstance(servers, Sequence) or isinstance(servers, (str, bytes)):
        raise ConfigLoadError("mcp_servers must be a list")
    return servers


def _tool_policy_from_mapping(item: Any, index: int) -> ToolPolicy:
    if not isinstance(item, Mapping):
        raise ConfigLoadError(f"policies[{index}] must be a mapping")
    data = dict(item)
    if "tool" in data and "tool_name" not in data:
        data["tool_name"] = data.pop("tool")
    if "agents" in data and "allowed_agents" not in data:
        data["allowed_agents"] = data.pop("agents")

    for name in ("allowed_agents", "denied_agents", "allowed_principals", "denied_principals"):
        if name in data:
            data[name] = _coerce_string_set(data[name])
    if "rate_limits" in data:
        data["rate_limits"] = tuple(_coerce_rate_limit_rule(item) for item in _coerce_list(data["rate_limits"]))
    if "metadata" in data and not isinstance(data["metadata"], Mapping):
        raise ConfigLoadError(f"policies[{index}].metadata must be a mapping")
    return ToolPolicy(**data)


def _mcp_server_from_mapping(item: Any, index: int) -> MCPServerConfig:
    if not isinstance(item, Mapping):
        raise ConfigLoadError(f"mcp_servers[{index}] must be a mapping")
    data = dict(item)
    if "id" in data and "server_id" not in data:
        data["server_id"] = data.pop("id")
    if "type" in data and "transport" not in data:
        data["transport"] = data.pop("type")
    if "timeout" in data and "timeout_ms" not in data:
        data["timeout_ms"] = data.pop("timeout")
    if isinstance(data.get("command"), str):
        data["command"] = (data["command"],)
    elif "command" in data:
        data["command"] = tuple(str(item) for item in _coerce_list(data["command"]))
    if "args" in data:
        data["args"] = tuple(str(item) for item in _coerce_list(data["args"]))
    for name in ("headers", "env", "metadata"):
        if name in data and not isinstance(data[name], Mapping):
            raise ConfigLoadError(f"mcp_servers[{index}].{name} must be a mapping")
    if "capabilities" in data:
        data["capabilities"] = tuple(str(item) for item in _coerce_list(data["capabilities"]))
    if "auth_type" in data:
        data["auth_type"] = AuthType(str(data["auth_type"]).lower())
    return MCPServerConfig(**data)


def _expand_mapping(values: Mapping[str, str], environ: Mapping[str, str]) -> dict[str, str]:
    return {str(key): _expand_env(str(value), environ) for key, value in values.items()}


def _expand_env(value: str, environ: Mapping[str, str]) -> str:
    expanded = value
    for key, env_value in environ.items():
        expanded = expanded.replace("${" + key + "}", env_value)
        expanded = expanded.replace("$" + key, env_value)
    return expanded


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _coerce_string_set(value: Any) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        return frozenset({value})
    if isinstance(value, Sequence):
        return frozenset(str(item) for item in value)
    raise ConfigLoadError("agent/principal list must be a string or list")


def _coerce_list(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    raise ConfigLoadError("rate_limits must be a list")


def _coerce_rate_limit_rule(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigLoadError("each rate_limits item must be a mapping")
    return dict(value)


def _parse_yaml_subset(text: str) -> Any:
    lines: list[tuple[int, str, int]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        if "\t" in raw[: len(raw) - len(raw.lstrip("\t "))]:
            raise ConfigLoadError(f"tabs are not supported in YAML indentation at line {lineno}")
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((indent, _strip_inline_comment(raw[indent:]).rstrip(), lineno))

    if not lines:
        return {}
    value, index = _parse_block(lines, 0, lines[0][0])
    if index != len(lines):
        raise ConfigLoadError(f"unexpected YAML content at line {lines[index][2]}")
    return value


def _parse_block(lines: list[tuple[int, str, int]], index: int, indent: int) -> tuple[Any, int]:
    current_indent, content, lineno = lines[index]
    if current_indent != indent:
        raise ConfigLoadError(f"invalid indentation at line {lineno}")
    if content.startswith("- "):
        return _parse_list(lines, index, indent)
    return _parse_mapping(lines, index, indent)


def _parse_mapping(lines: list[tuple[int, str, int]], index: int, indent: int) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    while index < len(lines):
        current_indent, content, lineno = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ConfigLoadError(f"unexpected indentation at line {lineno}")
        if content.startswith("- "):
            break
        key, has_value, raw_value = _split_key_value(content, lineno)
        index += 1
        if has_value:
            result[key] = _parse_scalar(raw_value)
            continue
        if index < len(lines) and lines[index][0] > indent:
            result[key], index = _parse_block(lines, index, lines[index][0])
        else:
            result[key] = None
    return result, index


def _parse_list(lines: list[tuple[int, str, int]], index: int, indent: int) -> tuple[list[Any], int]:
    result: list[Any] = []
    while index < len(lines):
        current_indent, content, lineno = lines[index]
        if current_indent < indent:
            break
        if current_indent != indent or not content.startswith("- "):
            break
        item_text = content[2:].strip()
        index += 1

        if not item_text:
            if index < len(lines) and lines[index][0] > indent:
                item, index = _parse_block(lines, index, lines[index][0])
            else:
                item = None
            result.append(item)
            continue

        if _looks_like_key_value(item_text):
            key, has_value, raw_value = _split_key_value(item_text, lineno)
            item_map: dict[str, Any] = {key: _parse_scalar(raw_value) if has_value else None}
            if index < len(lines) and lines[index][0] > indent:
                child, index = _parse_block(lines, index, lines[index][0])
                if isinstance(child, Mapping):
                    item_map.update(child)
                elif not has_value:
                    item_map[key] = child
                else:
                    raise ConfigLoadError(f"list item at line {lineno} cannot merge non-mapping child")
            result.append(item_map)
            continue

        result.append(_parse_scalar(item_text))
    return result, index


def _split_key_value(content: str, lineno: int) -> tuple[str, bool, str]:
    key, separator, value = content.partition(":")
    if not separator:
        raise ConfigLoadError(f"expected key: value at line {lineno}")
    key = key.strip()
    if not key:
        raise ConfigLoadError(f"empty mapping key at line {lineno}")
    raw_value = value.strip()
    return key, bool(raw_value), raw_value


def _looks_like_key_value(text: str) -> bool:
    key, separator, value = text.partition(":")
    return bool(separator and key.strip() and (not value or value.startswith(" ")))


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return _parse_inline_list(value[1:-1])
    if value.startswith("{") and value.endswith("}"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_inline_list(raw: str) -> list[Any]:
    if not raw.strip():
        return []
    return [_parse_scalar(item.strip()) for item in raw.split(",")]


def _strip_inline_comment(text: str) -> str:
    quote: str | None = None
    for index, char in enumerate(text):
        if char in {"'", '"'}:
            quote = None if quote == char else char if quote is None else quote
        if char == "#" and quote is None and (index == 0 or text[index - 1].isspace()):
            return text[:index]
    return text
