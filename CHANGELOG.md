# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Open-source release standardization: `__version__`, CHANGELOG, LICENSE, MANIFEST.in, CI/CD workflows, Makefile

## [0.1.0] - 2025-07-15

### Added
- Initial MCP tool harness release
- Tool registry with `server_id + tool_name + version` identity, schema hash, enable/disable, TTL cache
- Policy governance: `ToolPolicy`, Agent allowlist/denylist, risk levels L0-L3, high-risk approval
- YAML policy and MCP server configuration loading with hot-reload to Registry
- Runtime protection: timeout, rate limiting, circuit breaker, idempotency, schema validation
- Multi-dimensional rate limiting: tenant, agent, tool, server_tool, custom key templates
- MCP transport: stdio, SSE, Streamable HTTP, in-memory transport, tool discovery
- Audit and observability: call records, JSON Lines audit sink, metrics, trace helpers, `request_id` / `trace_id`
- Framework adapters: LangChain, LlamaIndex, OpenAI Agents SDK, AutoGen, CrewAI, Semantic Kernel
- FastAPI optional HTTP server with REST invoke and MCP JSON-RPC `/mcp` endpoint
- Web console: Tool, Chain, Metrics sidebar tabs
- DeepSeek Agent integration for testing tools
- Enterprise agent tracing and audit records
- Parameter deduplication, idempotency, and retries
- Policy circuit breaker configuration
- Global tool policy (`global_tool_policy`) support in YAML config
- MCP initialization, notifications, and protocol compatibility
