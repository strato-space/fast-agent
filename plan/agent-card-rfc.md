# AgentCard RFC (Draft)

## Summary
Define a text-first AgentCard format (md/yaml) that compiles to `AgentConfig` as the
Intermediate Representation. The loader validates attributes based on `type`, supports
loading a single file or an entire directory, and can parse multiple AgentCards from
one markdown file. Instructions live in the body after the frontmatter delimiter
(`---`).

## Minimal Attributes
- `type`: one of `agent`, `chain`, `parallel`, `evaluator_optimizer`, `router`,
  `orchestrator`, `iterative_planner`, `MAKER`
- `name`: unique agent name. If a file contains a single card and `name` is omitted,
  the agent name defaults to the filename without extension. If a file contains
  multiple cards, each card must specify `name`.
- `instruction`: may be stored either in the `instruction` attribute or in the
  markdown body after `---`. Both are supported; if both are present, they are
  concatenated with a newline.

## Attribute Sets
- All attributes defined by the decorator for a given `type` are permitted.
- The `type` determines the allowed attribute set.
- The loader enforces valid attributes and rejects unknown fields for that `type`.

## Schema Version
- `schema_version`: not needed for now.

## Instruction Location
- Instruction may be provided via the `instruction` attribute and/or in the body
  after the frontmatter delimiter (`---`).
- If both are present, they are concatenated with a newline (`instruction` attribute
  first, then body).
- System instruction may be prefixed with an optional `---SYSTEM` marker.

## Multi-Card Files (md)
- A single markdown file may contain multiple cards.
- Each card is a frontmatter block followed by body text.
- Cards are separated by a new `---` frontmatter block; body ends at the next block.
- `---SYSTEM` / `---USER` / `---ASSISTANT` are body markers and do not start a new card.
- In multi-card files, `name` is required on every card.

Example:
```md
---
type: agent
name: agent-name
---
You are a helpful assistant.
---
type: agent
name: agent-name2
---
You are a helpful assistant #2.
---
```

## History Preload (messages)
- Two options:
  - inline history inside the file body (e.g., `---USER` / `---ASSISTANT` blocks)
  - external history file referenced via `messages` attribute (path or list of paths)
- If both are present, message histories are merged.
- External `messages` can point to one or more prompt/history files (e.g., `.md`
  or `.json`).
- Intended for system instruction and user/assistant pairs and few-shot examples.

Example (single file):
```md
messages: ./history.md
```
Example (external `messages` in two files with md and json format):
```md
messages:
  - ./history.md
  - ./fewshot.json
```

## MCP Servers and Tool Filters (YAML)
- Keep the shape as close to existing decorator semantics as possible.
- `servers` entries may reference `fastagent.config.yaml` by name or include inline
  server config using the same schema (transport, url, auth, etc.).
- Attributes not explicitly set on an inline server entry inherit from
  `fastagent.config.yaml` when a matching server name exists.
- `tools` filters the allowed tools for that server (exact names or patterns). If
  omitted, the tool filter inherits from `fastagent.config.yaml` or defaults to
  "all tools" for that server.

Example:
```yaml
servers:
  - time:
      tools: [get_time]
  - github:
      tools: [search_*]
  - filesystem
  - myserver:
      transport: http
      url: http://localhost:8001/mcp
      tools: [search_*]
```

## Examples

### Agent with MCP servers, function tools, child agents, and hooks
Note: `---SYSTEM` is optional; if omitted or no, the body is treated as system instruction.
`---USER` and `---ASSISTANT` blocks are optional and start message history definition
that is loaded into the agent.

```md
---
type: agent
name: PMO-orchestrator
default: true
servers:  # mcp servers
  - time:
      tools: [get_time]       # exact tool
  - github:
      tools: [search_*]       # pattern
  - filesystem                # all tools
function_tools:
  - tools.py:local_summarize
  - tools.py:local_redact
agents:
  - NY-Project-Manager
  - London-Project-Manager
history_mode: scratch
max_parallel: 128
child_timeout_sec: 120
max_display_instances: 20
tool_hooks:
  - tools.py:audit_hook
  - tools.py:safety_guard
    match:
      server: github
      tool: search_*
  - hook: tools.py:rate_limit
    predicate: "ctx.tool_type in {'mcp','function'}"
---

---SYSTEM
Get reports. Always use one tool call per project/news.
Responsibilities: NY projects: [OpenAI, Fast-Agent, Anthropic].
London news: [Economics, Art, Culture].
Aggregate results and add a one-line PMO summary.

---USER
Provide the latest NY and London updates.

---ASSISTANT
Understood. I will fetch updates and return a concise PMO summary.
```

### Agent with preloaded history (inline blocks)
```md
---
type: agent
name: Report-Analyst
---

---SYSTEM
You are a concise analyst.

---USER
Summarize the dataset below in 3 bullet points.

---ASSISTANT
Use clear labels and highlight anomalies.

Analyze new reports and keep the output short and structured.
```

## Loading API
- `load_agents(path)` loads a file or a directory.
- Loading is immediate (no deferred mode).
- All loaded agents are tracked with a name and source file path. If a subsequent
  `load_agents(path)` call does not include a previously loaded agent from that
  path, the agent is removed.

## Hot Reload / Replace Behavior
- Internal representation keeps `file`/`url` origins and subscribes to updates
  by default.
- Provide CLI switch to choose reload mode:
  - `--reload-mode=hot-swap` (default): apply changes automatically on modify
  - `--reload-mode=explicit`: only reload via explicit command/tool call

## Tools Exposure (fast-agent-mcp)
Expose loader utilities via internal MCP tools, including update by name:
- `fast-agent-mcp.load_agents(path)`

## Appendix: Current History Preload (Code)
- `load_history_into_agent(agent, file_path)` loads a prompt/history file into
  `agent.message_history` without an LLM call.
  - Source: `src/fast_agent/mcp/prompts/prompt_load.py`
- `load_prompt(file)` returns `PromptMessageExtended` list; you can seed history via
  `agent.message_history.extend(messages)` or pass the list into `agent.generate(...)`.
  - Source: `src/fast_agent/mcp/prompts/prompt_load.py`
- TUI `/load_history` uses `load_history_into_agent`.
  - Source: `src/fast_agent/ui/interactive_prompt.py`
- CLI `--prompt-file` loads a prompt file and calls `agent.generate(...)`.
  - Source: `src/fast_agent/cli/commands/go.py`

## Open Question
- How should AgentCard reference Python callables for `function_tools` and
  `tool_hooks`? This is still unresolved; likely follow SKILL.md semantics.

## Appendix: AgentCard Samples
See `plan/agent-card-rfc-sample.md`.
