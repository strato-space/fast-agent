import asyncio

from fast_agent import FastAgent


def get_video_call_transcript(video_id: str) -> str:
    return "Assistant: Hi, how can I assist you today?\n\nCustomer: Hi, I wanted to ask you about last invoice I received..."


fast = FastAgent("Example Tool Use Application")


@fast.agent(
    name="main",
    function_tools=[get_video_call_transcript],
)
async def main() -> None:
    async with fast.run() as agent:
        await agent.default.generate(
            "What is the topic of the video call no.1234?",
        )


if __name__ == "__main__":
    asyncio.run(main())
