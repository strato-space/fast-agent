import pytest

from fast_agent.agents.tool_hooks import ToolHookContext
from fast_agent.agents.tool_runner import ToolRunnerHooks


def add_one(x: int) -> int:
    return x + 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_declarative_function_tools_and_hooks(fast_agent):
    fast = fast_agent
    events: list[str] = []
    hook_calls: list[tuple[str, str]] = []

    async def before_tool_call(runner, message):
        events.append("before_tool_call")

    async def after_tool_call(runner, message):
        events.append("after_tool_call")

    hooks = ToolRunnerHooks(
        before_tool_call=before_tool_call,
        after_tool_call=after_tool_call,
    )

    async def tool_hook(ctx: ToolHookContext, args, call_next):
        hook_calls.append((ctx.tool_source, ctx.tool_name))
        next_args = dict(args or {})
        next_args["x"] = 2
        return await call_next(next_args)

    @fast.agent(
        name="test",
        model="passthrough",
        function_tools=[add_one],
        tool_runner_hooks=hooks,
        tool_hooks=[tool_hook],
    )
    async def agent_function():
        async with fast.run() as app:
            assert app.test.tool_runner_hooks is hooks

            tools = await app.test.list_tools()
            assert any(tool.name == "add_one" for tool in tools.tools)

            result = await app.test.send('***CALL_TOOL add_one {"x": 1}')
            assert "3" in result
            assert events == ["before_tool_call", "after_tool_call"]
            assert hook_calls == [("function", "add_one")]

    await agent_function()
