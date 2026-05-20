# MCP Tool Harness

MCP Tool Harness 是一个轻量 Python SDK，用来把本地函数、内部 HTTP/RPC 能力包装成可注册、可限流、可校验、可被 Agent 调用的 MCP 风格工具。

它适合这些场景：

- 给 Agent 提供统一的工具注册和调用入口
- 快速验证内部工具是否适合暴露给 Agent
- 通过 HTTP 或最小 MCP JSON-RPC 暴露工具
- 适配 LangChain、LlamaIndex、OpenAI Agents SDK
- 使用 DeepSeek 测试工具调用链路

## 安装

从 GitHub 安装：

```bash
python -m pip install "git+https://github.com/zengiai/mcp-tool-harness.git"
```

从本地源码安装：

```bash
python -m pip install .
```

如果需要启动 HTTP 服务，再安装可选依赖：

```bash
python -m pip install fastapi uvicorn
```

核心 SDK 只依赖 Python 标准库。

## 快速开始

创建一个工具网关，注册一个加法工具，然后直接调用：

```python
from mcp_tool_harness.server import ToolGateway

gateway = ToolGateway(default_rate_limit_per_minute=120, default_timeout_ms=500)


def add(left: int, right: int) -> dict[str, int]:
    return {"value": left + right}


gateway.register_tool(
    "math.add",
    add,
    description="Add two integers",
    input_schema={
        "type": "object",
        "properties": {
            "left": {"type": "integer"},
            "right": {"type": "integer"},
        },
        "required": ["left", "right"],
    },
    timeout_ms=200,
)

response = gateway.invoke(
    "math.add",
    {"left": 1, "right": 2},
    principal="agent-a",
)

print(response.result)
```

输出：

```python
{"value": 3}
```

## 注册工具

`register_tool()` 接收一个 Python callable，并把它注册成可发现、可调用的工具。

```python
from mcp_tool_harness.server import ToolGateway

gateway = ToolGateway()


def echo(text: str) -> dict[str, str]:
    return {"text": text}


gateway.register_tool(
    "text.echo",
    echo,
    description="Echo input text",
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string"},
        },
        "required": ["text"],
    },
    tags=("demo",),
)

print(gateway.list_tools())
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `name` | 工具名，例如 `math.add`、`order.query` |
| `handler` | 实际执行工具逻辑的 Python 函数 |
| `description` | 给 Agent 看的工具描述 |
| `input_schema` | 工具入参 JSON Schema |
| `output_schema` | 工具出参 JSON Schema |
| `permissions` | 工具权限元数据 |
| `idempotent` | 标记工具是否幂等 |
| `rate_limit_per_minute` | 当前工具每分钟限流阈值 |
| `timeout_ms` | 当前工具超时时间 |
| `tags` | 工具标签 |

## 调用工具

同步调用：

```python
response = gateway.invoke(
    "text.echo",
    {"text": "hello"},
    principal="agent-a",
    request_id="req-001",
)

print(response.status)
print(response.result)
```

异步调用：

```python
response = await gateway.ainvoke(
    "text.echo",
    {"text": "hello"},
    principal="agent-a",
    request_id="req-001",
)
```

使用幂等 key：

```python
first = gateway.invoke(
    "text.echo",
    {"text": "hello"},
    principal="agent-a",
    idempotency_key="idem-001",
)

second = gateway.invoke(
    "text.echo",
    {"text": "hello"},
    principal="agent-a",
    idempotency_key="idem-001",
)

assert second.cached is True
assert second.result == first.result
```

## 访问控制

`ApprovalPolicy` 支持工具 allowlist、denylist 和需要人工审批的工具模式。

```python
from mcp_tool_harness.server import ApprovalPolicy, ToolGateway

policy = ApprovalPolicy(
    allowed_tools=("math.*", "text.*"),
    denied_tools=("danger.*",),
    approval_required_tools=("payment.refund",),
)

gateway = ToolGateway(approval_policy=policy)
```

当工具被拒绝时，网关会抛出 `PermissionDeniedError`：

```python
from mcp_tool_harness.server import PermissionDeniedError

try:
    gateway.invoke("danger.delete", principal="agent-a")
except PermissionDeniedError as exc:
    print(str(exc))
```

当工具需要审批时，网关会抛出 `ApprovalRequiredError`：

```python
from mcp_tool_harness.server import ApprovalRequiredError

try:
    gateway.invoke("payment.refund", {"order_id": "O-1001"}, principal="agent-a")
except ApprovalRequiredError as exc:
    print(exc.approval_id)
```

## 限流、超时和熔断

创建网关时设置默认限流和默认超时：

```python
gateway = ToolGateway(
    default_rate_limit_per_minute=60,
    default_timeout_ms=1_000,
    circuit_failure_threshold=3,
    circuit_recovery_seconds=30,
)
```

也可以给单个工具设置独立限流和超时：

```python
gateway.register_tool(
    "inventory.query",
    lambda sku_id: {"sku_id": sku_id, "stock": 100},
    input_schema={
        "type": "object",
        "properties": {"sku_id": {"type": "string"}},
        "required": ["sku_id"],
    },
    rate_limit_per_minute=600,
    timeout_ms=300,
)
```

## 暴露 HTTP 服务

创建 `app.py`：

```python
from mcp_tool_harness.server import ToolGateway, create_app

gateway = ToolGateway()
gateway.register_tool(
    "text.echo",
    lambda text: {"text": text},
    input_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)

app = create_app(gateway)
```

启动服务：

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

查看工具列表：

```bash
curl http://127.0.0.1:8000/tools
```

HTTP 调用工具：

```bash
curl -X POST http://127.0.0.1:8000/tools/text.echo/invoke \
  -H 'Content-Type: application/json' \
  -d '{
    "arguments": {"text": "hello"},
    "principal": "agent-a",
    "request_id": "req-001"
  }'
```

## MCP JSON-RPC 调用

同一个 HTTP 服务也提供 `/mcp` 入口。

列出工具：

```bash
curl -X POST http://127.0.0.1:8000/mcp \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": "list-001",
    "method": "tools/list"
  }'
```

调用工具：

```bash
curl -X POST http://127.0.0.1:8000/mcp \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": "call-001",
    "method": "tools/call",
    "params": {
      "name": "text.echo",
      "arguments": {"text": "hello"},
      "principal": "agent-a"
    }
  }'
```

## 包装 HTTP 或 RPC 能力

把内部 HTTP/RPC client 包成普通 Python 函数后注册即可。

```python
from mcp_tool_harness.server import ToolGateway


class InventoryClient:
    def query(self, sku_id: str) -> dict[str, object]:
        return {
            "sku_id": sku_id,
            "stock": 100,
            "available": True,
        }


client = InventoryClient()
gateway = ToolGateway(default_timeout_ms=500)

gateway.register_tool(
    "inventory.query",
    lambda sku_id: client.query(sku_id),
    description="Query inventory by SKU id",
    input_schema={
        "type": "object",
        "properties": {"sku_id": {"type": "string"}},
        "required": ["sku_id"],
    },
    timeout_ms=300,
)

result = gateway.invoke(
    "inventory.query",
    {"sku_id": "SKU-1001"},
    principal="inventory-agent",
)

print(result.result)
```

## 使用 DeepSeek 测试工具

先准备一个 gateway：

```python
from mcp_tool_harness.agent import create_deepseek_agent
from mcp_tool_harness.server import ToolGateway

gateway = ToolGateway(default_rate_limit_per_minute=120, default_timeout_ms=2_000)

gateway.register_tool(
    "math.add",
    lambda left, right: {"value": left + right},
    description="Add two integers",
    input_schema={
        "type": "object",
        "properties": {
            "left": {"type": "integer"},
            "right": {"type": "integer"},
        },
        "required": ["left", "right"],
    },
)

agent = create_deepseek_agent(
    gateway,
    base_url="https://api.deepseek.com",
    api_key="<your-api-key>",
    model="deepseek-chat",
)

result = agent.run("用工具计算 12 + 30")
print(result.content)
print([item.to_dict() for item in result.tool_invocations])
```

也可以直接使用内置 demo 工具运行命令：

```bash
export DEEPSEEK_API_KEY="<your-api-key>"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_MODEL="deepseek-chat"

python -m mcp_tool_harness.agent.deepseek "用工具计算 12 + 30" --show-tool-results
```

## 适配 LangChain

```python
from mcp_tool_harness.adapters.langchain import to_langchain_tool
from mcp_tool_harness.mcp.discovery import ToolSpec


class HarnessClient:
    def __init__(self, gateway):
        self.gateway = gateway

    def call_tool(self, name, arguments):
        response = self.gateway.invoke(name, arguments, principal="langchain-agent")
        return {"structuredContent": response.result}


spec = ToolSpec(
    name="text.echo",
    description="Echo input text",
    input_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)

tool = to_langchain_tool(HarnessClient(gateway), spec)
```

## 适配 LlamaIndex

```python
from mcp_tool_harness.adapters.llamaindex import to_llamaindex_tool
from mcp_tool_harness.mcp.discovery import ToolSpec

spec = ToolSpec(
    name="text.echo",
    description="Echo input text",
    input_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)

tool = to_llamaindex_tool(HarnessClient(gateway), spec)
```

## 适配 OpenAI Agents SDK

```python
from mcp_tool_harness.adapters.openai_agents import (
    to_openai_agents_tool,
    to_openai_tool_schema,
)
from mcp_tool_harness.mcp.discovery import ToolSpec

spec = ToolSpec(
    name="text.echo",
    description="Echo input text",
    input_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)

tool_schema = to_openai_tool_schema(spec)
tool = to_openai_agents_tool(HarnessClient(gateway), spec)
```

## 常见异常

| 异常 | 触发场景 |
| --- | --- |
| `ToolNotFoundError` | 工具没有注册 |
| `ToolInputValidationError` | 参数不满足 `input_schema` |
| `PermissionDeniedError` | 工具被访问策略拒绝 |
| `ApprovalRequiredError` | 工具需要审批 |
| `RateLimitExceededError` | 触发限流 |
| `CircuitOpenError` | 工具连续失败后熔断打开 |
| `IdempotencyConflictError` | 同一个幂等 key 被不同请求复用 |
| `ToolTimeoutError` | 工具执行超时 |
| `ToolExecutionError` | 工具 handler 抛出异常 |
