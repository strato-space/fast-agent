from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Sequence

from mcp.types import CallToolResult

ToolCallArgs = dict[str, Any] | None
ToolCallFn = Callable[[ToolCallArgs], Awaitable[CallToolResult]]
ToolHookFn = Callable[["ToolHookContext", ToolCallArgs, ToolCallFn], Awaitable[CallToolResult]]


@dataclass(frozen=True)
class ToolHookContext:
    agent_name: str
    server_name: str | None
    tool_name: str
    tool_source: Literal["mcp", "function", "agent", "runtime"]
    tool_use_id: str | None
    correlation_id: str | None
    original_tool_func: ToolCallFn


async def run_tool_with_hooks(
    hooks: Sequence[ToolHookFn],
    context: ToolHookContext,
    arguments: ToolCallArgs,
) -> CallToolResult:
    if not hooks:
        return await context.original_tool_func(arguments)

    async def call_at(index: int, args: ToolCallArgs) -> CallToolResult:
        if index >= len(hooks):
            return await context.original_tool_func(args)

        hook = hooks[index]

        async def call_next(next_args: ToolCallArgs) -> CallToolResult:
            return await call_at(index + 1, next_args)

        return await hook(context, args, call_next)

    return await call_at(0, arguments)
