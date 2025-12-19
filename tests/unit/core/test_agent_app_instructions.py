import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.core.agent_app import AgentApp
from fast_agent.types import RequestParams


class _DummyLLM:
    def __init__(self) -> None:
        self.default_request_params = RequestParams(systemPrompt="llm-old")
        self.instruction = "llm-old"


class _DummyAgent:
    def __init__(self) -> None:
        self.config = AgentConfig(name="worker", instruction="agent-old")
        self.instruction = self.config.instruction
        self._default_request_params = self.config.default_request_params
        self._llm = _DummyLLM()
        self._llm_attach_kwargs = {"request_params": RequestParams(systemPrompt="attach-old")}
        self.initialized = True
        self.apply_called = False

    async def _apply_instruction_templates(self) -> None:
        self.apply_called = True
        self.instruction = self.instruction.replace("{{serverInstructions}}", "<server/>")
        self._default_request_params.systemPrompt = self.instruction


@pytest.mark.asyncio
async def test_write_instructions_updates_template_and_effective_prompt():
    agent = _DummyAgent()
    app = AgentApp({"worker": agent})

    template = "AA{{file_silent:missing.txt}}BB {{serverInstructions}}"
    resolved = await app.write_instructions(template, agent_name="worker")

    assert app.read_instructions("worker") == template
    assert resolved == "AABB <server/>"
    assert agent.instruction == "AABB <server/>"
    assert agent._default_request_params.systemPrompt == "AABB <server/>"
    assert agent.config.default_request_params.systemPrompt == "AABB <server/>"
    assert agent._llm.default_request_params.systemPrompt == "AABB <server/>"
    assert agent._llm.instruction == "AABB <server/>"
    assert agent._llm_attach_kwargs["request_params"].systemPrompt == "AABB <server/>"
    assert agent.apply_called is True

