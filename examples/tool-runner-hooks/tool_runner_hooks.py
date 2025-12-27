import asyncio

from fast_agent import FastAgent
from fast_agent.agents.tool_runner import ToolRunnerHooks
from fast_agent.types import PromptMessageExtended


def get_video_call_transcript(video_id: str) -> str:
    return "Assistant: Hi, how can I assist you today?\n\nCustomer: Hi, I wanted to ask you about last invoice I received..."


async def add_style_hint(runner, messages: list[PromptMessageExtended]) -> None:
    if runner.iteration == 0:
        runner.append_messages("Keep the answer to one short sentence.")


async def log_tool_result(runner, message: PromptMessageExtended) -> None:
    if message.tool_results:
        tool_names = ", ".join(message.tool_results.keys())
        print(f"[hook] tool results received: {tool_names}")


hooks = ToolRunnerHooks(
    before_llm_call=add_style_hint,
    after_tool_call=log_tool_result,
)


fast = FastAgent("Example Tool Use Application (Hooks)")


@fast.agent(
    name="main",
    function_tools=[get_video_call_transcript],
    tool_runner_hooks=hooks,
)
async def main() -> None:
    async with fast.run() as agent:
        await agent.default.generate(
            "What is the topic of the video call no.1234?",
        )
        await agent.interactive()


if __name__ == "__main__":
    asyncio.run(main())
