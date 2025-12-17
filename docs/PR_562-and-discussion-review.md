# PR #562 + Discussion Review (fast-agent)

## MCP in fast-agent (What it is + what you have)

- MCP = a standard way for an “LLM client” (fast-agent) to talk to “MCP servers” that provide **tools**, **resources**, **prompts**, plus newer features like **sampling** and **elicitations**.
- fast-agent as an MCP *client*: `fast-agent/src/fast_agent/agents/mcp_agent.py:44` + `fast-agent/src/fast_agent/mcp/mcp_aggregator.py:1383` (tool routing, permissions, progress hooks) + `fast-agent/src/fast_agent/mcp/mcp_agent_client_session.py:1` (roots, sampling callback, elicitation callback).
- MCP “features” implemented here (high level):
  - **Tools**: discovery + namespacing + execution via aggregator (`fast-agent/src/fast_agent/mcp/mcp_aggregator.py:1383`).
  - **Progress notifications**: forwarded into a handler (ACP uses this for UI) (`fast-agent/src/fast_agent/mcp/mcp_aggregator.py:1466`).
  - **Permissions**: abstracted via a handler; ACP plugs in a real permission adapter (`fast-agent/src/fast_agent/mcp/mcp_aggregator.py:1409`).
  - **Roots**: exposed to servers via `list_roots` callback (`fast-agent/src/fast_agent/mcp/mcp_agent_client_session.py:38`).
  - **Elicitations**: implemented via forms/auto-cancel handlers (`fast-agent/src/fast_agent/mcp/elicitation_handlers.py:1`).
  - **Sampling**: implemented, but currently “no tools in sampling” (`fast-agent/src/fast_agent/mcp/sampling.py:24`).

## ACP (Why it’s the best base for a web UI)

- ACP is a *client ↔ agent* protocol (separate from MCP). fast-agent already implements an ACP server with:
  - streaming text updates,
  - structured tool-call lifecycle events,
  - permission prompts,
  - optional terminal/filesystem “runtime” injection.
- The wiring is already there in `fast-agent/src/fast_agent/acp/server/agent_acp_server.py:540` (registers tool progress handler, tool stream listener, and permission handler per session). This is exactly what a web UI needs for “live execution panel + logs”.

## Contributor idea: `read_instructions` / `write_instructions`

- It already exists and is tested:
  - API: `fast-agent/src/fast_agent/core/agent_app.py:60` and `fast-agent/src/fast_agent/core/agent_app.py:77`.
  - Test: `fast-agent/tests/unit/core/test_agent_app_instructions.py:1`.
- Why it feels “complex”: changing an instruction isn’t one field; it must stay consistent across `AgentConfig`, `RequestParams.systemPrompt`, attached LLM defaults, and clone/spawn settings (otherwise children/tool clones keep old prompts). The current implementation correctly synchronizes those layers.

## PR review: upstream PR #562 (branch `pr-562`)

- What it’s trying to enable: SEP-1577 “Sampling With Tools” (MCP spec change) — tools + toolChoice inside sampling requests, and more generally a reusable “tool loop runner”.
- What PR #562 changes (diff summary):
  - Adds `src/fast_agent/agents/tool_runner.py` (in PR) to centralize the tool loop.
  - Updates tool execution to run multiple tool calls in parallel by default, with a kill-switch constant (`FORCE_SEQUENTIAL_TOOL_CALLS`).
- Main correctness bug to fix before merge:
  - In `pr-562:src/fast_agent/agents/tool_runner.py:124`, `_pending_tool_response` is cached and **never reset**. If the LLM does tools in *multiple rounds* (tool_use → tool results → tool_use again), the second round will incorrectly reuse the first cached tool response and skip executing new tools.
  - Minimal fix: reset `_pending_tool_response = None` whenever a new `_pending_tool_request` is set (at `pr-562:src/fast_agent/agents/tool_runner.py:109`) and add a regression test with two tool-use rounds.
- Design risks to consider (not blockers, but important):
  - Parallel tool execution is great for latency, but unsafe for some tools (human input, terminal) and possibly some MCP server/session implementations; consider per-tool/per-server gating instead of only a global constant.

## Recommended path for the “full web interface”

- Fastest/cleanest: build a web app that runs an **ACP client** backend which spawns `fast-agent serve --transport acp` and relays ACP JSON-RPC + notifications over WebSocket to the browser. You immediately get streaming + tool timeline + permissions without inventing a new protocol.
- Then add instruction editing in the UI by calling the existing `AgentApp.write_instructions()` API (`fast-agent/src/fast_agent/core/agent_app.py:77`) and persisting it to the instruction file/config if you want it to survive restarts.

