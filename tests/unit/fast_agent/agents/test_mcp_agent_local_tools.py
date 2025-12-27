from typing import TYPE_CHECKING, Any

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.agents.tool_hooks import ToolHookContext
from fast_agent.context import Context
from fast_agent.mcp.helpers.content_helpers import text_content

if TYPE_CHECKING:
    from mcp.types import CallToolResult


def _make_agent_config() -> AgentConfig:
    return AgentConfig(name="test-agent", instruction="do things", servers=[])


@pytest.mark.asyncio
async def test_local_tools_listed_and_callable() -> None:
    calls: list[dict[str, Any]] = []

    def sample_tool(video_id: str) -> str:
        calls.append({"video_id": video_id})
        return f"transcript for {video_id}"

    config = _make_agent_config()
    context = Context()

    class LocalToolAgent(McpAgent):
        def __init__(self) -> None:
            super().__init__(
                config=config,
                connection_persistence=False,
                context=context,
                tools=[sample_tool],
            )

    agent = LocalToolAgent()

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "sample_tool" in tool_names

    result: CallToolResult = await agent.call_tool("sample_tool", {"video_id": "1234"})
    assert not result.isError
    assert calls == [{"video_id": "1234"}]
    assert [block.text for block in result.content or []] == ["transcript for 1234"]

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_tool_hooks_apply_to_local_tools() -> None:
    calls: list[dict[str, Any]] = []
    hook_seen: list[tuple[str, str | None, str]] = []

    def sample_tool(video_id: str) -> str:
        calls.append({"video_id": video_id})
        return f"transcript for {video_id}"

    config = _make_agent_config()
    context = Context()

    class LocalToolAgent(McpAgent):
        def __init__(self) -> None:
            super().__init__(
                config=config,
                connection_persistence=False,
                context=context,
                tools=[sample_tool],
            )

    async def hook_one(ctx: ToolHookContext, args, call_next):
        hook_seen.append((ctx.tool_source, ctx.server_name, ctx.tool_name))
        next_args = dict(args or {})
        next_args["video_id"] = "9999"
        return await call_next(next_args)

    async def hook_two(ctx: ToolHookContext, args, call_next):
        result = await call_next(args)
        result.content.append(text_content("[hooked]"))
        return result

    agent = LocalToolAgent()
    agent.tool_hooks = [hook_one, hook_two]

    result: CallToolResult = await agent.call_tool("sample_tool", {"video_id": "1234"})
    assert not result.isError
    assert calls == [{"video_id": "9999"}]
    assert [block.text for block in result.content or []] == [
        "transcript for 9999",
        "[hooked]",
    ]
    assert hook_seen == [("function", None, "sample_tool")]

    await agent._aggregator.close()
