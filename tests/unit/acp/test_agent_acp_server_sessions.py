from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from acp.schema import SessionModeState

from fast_agent.acp.server.agent_acp_server import ACPSessionState, AgentACPServer
from fast_agent.core.agent_app import AgentApp
from fast_agent.core.fastagent import AgentInstance
from fast_agent.session.session_manager import SessionInfo

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol


class _Agent:
    instruction = ""
    acp_commands: dict[str, object] = {}

    def __init__(self) -> None:
        self.config = SimpleNamespace(default=False)


def _build_instance(agent_names: list[str]) -> AgentInstance:
    agents = {
        name: cast("AgentProtocol", _Agent())
        for name in agent_names
    }
    return AgentInstance(
        app=AgentApp(agents),
        agents=agents,
        registry_version=0,
    )


def _build_server(instance: AgentInstance) -> AgentACPServer:
    async def create_instance() -> AgentInstance:
        return instance

    async def dispose_instance(_instance: AgentInstance) -> None:
        return None

    return AgentACPServer(
        primary_instance=instance,
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        instance_scope="shared",
        server_name="test",
        permissions_enabled=False,
    )


@pytest.mark.asyncio
async def test_load_session_falls_back_when_primary_agent_was_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_instance = _build_instance(["main"])
    server = _build_server(primary_instance)
    refreshed_instance = _build_instance(["renamed"])
    session_state = ACPSessionState(
        session_id="s-1",
        instance=refreshed_instance,
    )

    async def fake_initialize_session_state(
        session_id: str,
        *,
        cwd: str,
        mcp_servers: list[Any],
    ) -> tuple[ACPSessionState, SessionModeState]:
        del session_id, cwd, mcp_servers
        return session_state, SessionModeState(available_modes=[], current_mode_id="main")

    resume_calls: list[str | None] = []

    def fake_get_session_manager(*, cwd: Any = None) -> Any:
        del cwd

        class _Manager:
            def load_session(self, name: str) -> Any:
                del name
                return SimpleNamespace(info=SimpleNamespace(metadata={}))

            def resume_session_agents(
                self,
                agents: Any,
                name: str | None = None,
                fallback_agent_name: str | None = None,
            ) -> Any:
                del agents, name
                resume_calls.append(fallback_agent_name)
                return SimpleNamespace(
                    session=SimpleNamespace(),
                    loaded={},
                    missing_agents=["main"],
                    usage_notices=[],
                )

        return _Manager()

    monkeypatch.setattr(server, "_initialize_session_state", fake_initialize_session_state)
    monkeypatch.setattr(
        "fast_agent.acp.server.agent_acp_server.get_session_manager",
        fake_get_session_manager,
    )

    response = await server.load_session(
        cwd=".",
        session_id="s-1",
        mcp_servers=[],
    )

    assert response is not None
    assert response.modes is not None
    assert resume_calls == ["renamed"]
    assert session_state.current_agent_name == "renamed"
    assert response.modes.current_mode_id == "renamed"


@pytest.mark.asyncio
async def test_list_sessions_keeps_legacy_sessions_when_cwd_matches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    server = _build_server(_build_instance(["main"]))
    workspace = tmp_path / "workspace"
    legacy_sessions_dir = workspace / ".fast-agent" / "sessions"
    now = datetime.now()

    legacy_session = SessionInfo(
        name="legacy",
        created_at=now,
        last_activity=now,
        metadata={},
    )
    explicit_session = SessionInfo(
        name="explicit",
        created_at=now,
        last_activity=now,
        metadata={"cwd": str(workspace.resolve())},
    )
    other_session = SessionInfo(
        name="other",
        created_at=now,
        last_activity=now,
        metadata={"cwd": str((tmp_path / "other").resolve())},
    )

    class _Manager:
        base_dir = legacy_sessions_dir

        def list_sessions(self) -> list[SessionInfo]:
            return [legacy_session, explicit_session, other_session]

    monkeypatch.setattr(
        "fast_agent.acp.server.agent_acp_server.get_session_manager",
        lambda: _Manager(),
    )

    response = await server.list_sessions(cwd=str(workspace))

    assert [session.session_id for session in response.sessions] == ["legacy", "explicit"]
    assert [session.cwd for session in response.sessions] == [
        str(workspace.resolve()),
        str(workspace.resolve()),
    ]


@pytest.mark.asyncio
async def test_list_sessions_uses_manager_workspace_for_legacy_sessions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    server = _build_server(_build_instance(["main"]))
    workspace = tmp_path / "workspace"
    env_root = tmp_path / "custom-env"
    custom_sessions_dir = env_root / "sessions"
    now = datetime.now()

    legacy_session = SessionInfo(
        name="legacy",
        created_at=now,
        last_activity=now,
        metadata={},
    )
    other_session = SessionInfo(
        name="other",
        created_at=now,
        last_activity=now,
        metadata={"cwd": str((tmp_path / "other").resolve())},
    )

    class _Manager:
        workspace_dir = workspace
        base_dir = custom_sessions_dir

        def list_sessions(self) -> list[SessionInfo]:
            return [legacy_session, other_session]

    monkeypatch.setattr(
        "fast_agent.acp.server.agent_acp_server.get_session_manager",
        lambda: _Manager(),
    )

    response = await server.list_sessions(cwd=str(workspace))

    assert [session.session_id for session in response.sessions] == ["legacy"]
    assert [session.cwd for session in response.sessions] == [str(workspace.resolve())]
