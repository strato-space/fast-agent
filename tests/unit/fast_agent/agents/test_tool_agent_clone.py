import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.agents.tool_hooks import ToolHookContext
from fast_agent.agents.tool_runner import ToolRunnerHooks


def ping() -> str:
    return "pong"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_agent_clone_preserves_tools_and_hooks():
    hooks = ToolRunnerHooks()

    async def tool_hook(ctx: ToolHookContext, args, call_next):
        return await call_next(args)

    tool_hooks = [tool_hook]
    agent = ToolAgent(
        AgentConfig("test"),
        tools=[ping],
        tool_runner_hooks=hooks,
        tool_hooks=tool_hooks,
    )

    clone = await agent.spawn_detached_instance(name="clone")

    tools = await clone.list_tools()
    assert any(tool.name == "ping" for tool in tools.tools)
    assert clone.tool_runner_hooks is hooks
    assert clone.tool_hooks == tool_hooks
