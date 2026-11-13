"""
Claude provider: wraps the Claude Agent SDK.

The SDK supplies the agentic loop. Built-in tools (Read/Bash/Grep/Glob/
WebFetch/Write) map to the SDK's native implementations; any other Tool is
exposed to the model through an in-process MCP server.

Vertex AI / Bedrock are auto-detected by the SDK from environment variables
(CLAUDE_CODE_USE_VERTEX, ANTHROPIC_VERTEX_PROJECT_ID, CLOUD_ML_REGION, ...), so
no explicit configuration is needed here. A custom endpoint (gateway, proxy,
...) comes from base_url, else ANTHROPIC_BASE_URL in the environment.
"""

from typing import Any

from ..reporting import Reporter
from ..result import RunRequest, RunResult, Usage
from ..tools.base import Tool, ToolContext
from .base import Provider

# Name of the in-process MCP server carrying custom tools. The SDK exposes its
# tools to the model as mcp__<server>__<tool>.
MCP_SERVER_NAME = "harness"
_MCP_PREFIX = f"mcp__{MCP_SERVER_NAME}__"


class ClaudeProvider(Provider):
    name = "claude"

    def __init__(self, permission_mode: str = "acceptEdits",
                 options: dict[str, Any] | None = None, base_url: str | None = None):
        """
        Args:
            permission_mode: Claude Agent SDK permission mode.
            options: Extra ClaudeAgentOptions fields, passed through verbatim
                (e.g. {"setting_sources": []} to ignore ~/.claude settings).
            base_url: Custom Anthropic API endpoint. Defaults to
                ANTHROPIC_BASE_URL, which the SDK's CLI inherits on its own.
        """
        self.permission_mode = permission_mode
        self.options = options or {}
        self.base_url = base_url

    async def run(self, request: RunRequest, reporter: Reporter) -> RunResult:
        try:
            from claude_agent_sdk import (
                AssistantMessage,
                ClaudeAgentOptions,
                ResultMessage,
                TextBlock,
                ToolResultBlock,
                ToolUseBlock,
                UserMessage,
                query,
            )
        except ImportError as e:
            raise RuntimeError(
                "The 'claude-agent-sdk' package is required for the claude "
                "provider. Install it with: pip install 'omniharness[claude]'"
            ) from e

        native = [t.name for t in request.tools if t.claude_native]
        custom = [t for t in request.tools if not t.claude_native]

        option_values: dict[str, Any] = dict(
            model=request.model,
            # `tools` restricts which built-ins exist at all; `allowed_tools`
            # auto-approves them.
            tools=native,
            allowed_tools=native + [_MCP_PREFIX + t.name for t in custom],
            permission_mode=self.permission_mode,
            cwd=request.work_dir,
            add_dirs=list(request.add_dirs),
            env=dict(request.env),
            system_prompt=request.system_prompt or None,
            max_turns=request.max_turns,
        )
        if custom:
            ctx = ToolContext(work_dir=request.work_dir, env=dict(request.env))
            option_values["mcp_servers"] = {MCP_SERVER_NAME: _mcp_server(custom, ctx)}
        if request.effort is not None:
            option_values["effort"] = request.effort
        if self.base_url:
            # The SDK starts its CLI with the inherited environment plus `env`.
            option_values["env"]["ANTHROPIC_BASE_URL"] = self.base_url
        option_values.update(self.options)
        options = ClaudeAgentOptions(**option_values)

        result = RunResult(success=False, subtype="error", provider=self.name,
                           model=request.model)
        # tool_use_id -> tool name, to label results.
        pending: dict[str, str] = {}
        last_text = ""

        async for message in query(prompt=request.prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text:
                        last_text = block.text
                        reporter.on_text(block.text)
                    elif isinstance(block, ToolUseBlock):
                        name = _strip_prefix(block.name)
                        pending[block.id] = name
                        reporter.on_tool_call(name, block.input or {})

            elif isinstance(message, UserMessage) and isinstance(message.content, list):
                for block in message.content:
                    if isinstance(block, ToolResultBlock):
                        reporter.on_tool_result(
                            pending.pop(block.tool_use_id, "?"),
                            _result_text(block.content),
                            bool(block.is_error),
                        )

            elif isinstance(message, ResultMessage):
                result.subtype = message.subtype
                result.success = message.subtype == "success" and not message.is_error
                result.final_text = message.result or last_text
                result.usage = _usage(message.usage)
                result.model_usage = message.model_usage
                result.total_cost_usd = message.total_cost_usd
                result.session_id = message.session_id
                result.num_turns = message.num_turns

        return result


def _mcp_server(tools: list[Tool], ctx: ToolContext):
    from claude_agent_sdk import SdkMcpTool, create_sdk_mcp_server

    def make_handler(t: Tool):
        async def handler(args: dict[str, Any]) -> dict[str, Any]:
            text = await t.ainvoke(args, ctx)
            return {"content": [{"type": "text", "text": text}]}

        return handler

    sdk_tools = [
        SdkMcpTool(
            name=t.name,
            description=t.description,
            input_schema=t.parameters,
            handler=make_handler(t),
        )
        for t in tools
    ]
    return create_sdk_mcp_server(name=MCP_SERVER_NAME, tools=sdk_tools)


def _strip_prefix(name: str) -> str:
    return name.removeprefix(_MCP_PREFIX)


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _usage(raw: dict[str, Any] | None) -> Usage:
    raw = raw or {}
    return Usage(
        input_tokens=raw.get("input_tokens", 0) or 0,
        cache_read_input_tokens=raw.get("cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=raw.get("cache_creation_input_tokens", 0) or 0,
        output_tokens=raw.get("output_tokens", 0) or 0,
        reasoning_tokens=(raw.get("output_tokens_details") or {}).get("thinking_tokens", 0) or 0,
    )
