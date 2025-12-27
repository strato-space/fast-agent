"""Declarative tool hooks example.

Shows how to mix MCP servers, local function tools, agents-as-tools, and
nginx-style tool hooks in one agent declaration.

Related examples:
- Function tools: examples/new-api/simple_llm.py
- Tool runner hooks: examples/tool-runner-hooks/tool_runner_hooks.py
- Agents-as-tools: examples/workflows/agents_as_tools_simple.py
"""

import asyncio

from fast_agent import FastAgent
from fast_agent.mcp.helpers.content_helpers import text_content
from mcp.types import CallToolResult


fast = FastAgent("Declarative Tool Hooks")


def add_one(x: int) -> int:
    return x + 1


async def audit_hook(ctx, args, call_next):
    # before: enforce limits
    if ctx.tool_name.endswith("add_one"):
        args = dict(args or {})
        args["x"] = min(int(args.get("x", 0)), 10)

    # instead: block unsafe tools
    if ctx.tool_source == "runtime" and ctx.tool_name == "shell.execute":
        return CallToolResult(isError=True, content=[text_content("blocked")])

    # call original tool
    result = await call_next(args)

    # after: log or modify result
    result.content.append(text_content("[audit]"))
    return result


@fast.agent(
    name="NY-Project-Manager",
    instruction="Return NY time + timezone, plus a one-line project status.",
    servers=["time"],
)
@fast.agent(
    name="London-Project-Manager",
    instruction="Return London time + timezone, plus a one-line news update.",
    servers=["time"],
)
@fast.agent(
    name="PMO-orchestrator",
    instruction="Aggregate reports and summarize in one line.",
    agents=["NY-Project-Manager", "London-Project-Manager"],
    servers=["time"],
    tools={"time": ["get_time"]},
    function_tools=[add_one],
    tool_hooks=[audit_hook],
    default=True,
)
async def main() -> None:
    async with fast.run() as agent:
        await agent("Run PMO report and add 1 to 3")


if __name__ == "__main__":
    asyncio.run(main())
