# AgentCard RFC (Draft)

## Summary
AgentCard is a **text-first** format (`.md` / `.yaml`) that compiles into `AgentConfig`.
A loader validates fields based on `type`, supports loading a single file or a directory,
and supports **multiple AgentCards per file** ("bundles").

The design goal is: **easy to write by humans, unambiguous to parse by machines**.

## Goals
- One canonical IR: `AgentConfig`.
- Strong validation: reject unknown fields for the given `type`.
- Support these authoring styles:
  - YAML-only cards (`instruction` as a YAML literal block)
  - Markdown cards: YAML frontmatter + body (instruction/history)
- Support "agent bundles": multiple cards in one file with unambiguous boundaries.

## Non-goals (for now)
- Cross-file imports/includes (beyond `messages:` referencing external history files).
- A rich schema migration framework.

---

## Terminology
- **Card**: one AgentCard definition (`type` + attributes + instruction/history).
- **Bundle**: one file containing multiple cards.
- **Frontmatter**: YAML header delimited by `---` lines.
- **Body**: markdown text following the frontmatter; used for instruction and/or message history.

---

## Minimal Attributes
- `type`: one of `agent`, `chain`, `parallel`, `evaluator_optimizer`, `router`, `orchestrator`,
  `iterative_planner`, `MAKER`
- `name`: unique card name within a load-set.
  - If a file contains a **single** card and `name` is omitted, it defaults to the filename (no extension).
  - If a file contains **multiple** cards, each card **must** specify `name`.
- `instruction`: may be stored either in the `instruction` attribute or in the body.
  - Both are supported; if both are present, they are concatenated with a newline
    (`instruction` attribute first, then body-derived system instruction).

## Attribute Sets
- All attributes defined by the decorator for a given `type` are permitted.
- `type` determines the allowed attribute set.
- The loader enforces valid attributes and rejects unknown fields for that `type`.

## Schema Version
- `schema_version`: optional.
  - If present, must be an integer.
  - Loader should default to `1` when omitted.
  - Parser/loader must remain backwards-compatible within a major series when feasible.

---

## Supported File Formats

### 1) YAML Card (`.yaml` / `.yml`)
A YAML card is a single YAML document whose keys map directly to the `AgentConfig` schema.
Use `instruction: |` for multiline prompts.

Example:
```yaml
type: agent
name: sizer
instruction: |
  Given an object, respond only with an estimate of its size.
```

### 2) YAML Bundle (multi-doc YAML stream)
A YAML file may contain **multiple YAML documents**, separated by YAML document markers.
Each document is one card.

Example:
```yaml
---
type: agent
name: sizer
instruction: |
  Given an object, respond only with an estimate of its size.
---
type: agent
name: greeter
instruction: |
  Respond cheerfully.
```

### 3) Markdown Card (`.md`)
A Markdown card is **YAML frontmatter** followed by an optional body.

Example:
```md
---
type: agent
name: sizer
---
Given an object, respond only with an estimate of its size.
```

### 4) Markdown Bundle (multi-card `.md`)
A single markdown file may contain multiple cards. Each card is:
- a frontmatter block (YAML) delimited by `---` lines
- followed by a body (markdown) until the next card

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
```

---

## Card Boundary and Parsing Rules (Markdown)

### Frontmatter definition
- Frontmatter **starts** at a line that is exactly `---` (column 0, no leading/trailing spaces).
- Frontmatter **ends** at the next line that is exactly `---`.
- The text between them is parsed as YAML.

### What counts as a new card?
In a markdown bundle, not every `---` line is a card boundary (markdown uses `---` as a horizontal rule).
To avoid ambiguity, the loader uses this rule:

> A `---` line starts a new card **only if** the subsequent frontmatter YAML parses successfully and contains a `type` key.

Operationally:
1. Scan for a `---` line.
2. Parse YAML until the next closing `---`.
3. If YAML contains `type`, treat it as a card start.
4. Otherwise treat it as normal markdown content (e.g., a horizontal rule) and continue scanning.

This makes boundaries deterministic even when the body contains `---`.

### Body range
- The card body begins immediately after the closing frontmatter delimiter.
- The card body ends at the start of the next valid card (as defined above) or end-of-file.

### Name rules in bundles
- In multi-card markdown files, `name` is required on every card.
- Loader must reject duplicate `name`s within the load-set.

---

## Instruction and Message History in Body

### Body as system instruction
- If the body contains no explicit message blocks, the entire body is treated as **system instruction**.
- If the body contains message blocks, the blocks define message history, and any non-block text
  **before the first block** is treated as additional system instruction.

### Message blocks (inline history)
The body may include blocks that seed message history.
Block headers must appear at column 0:
- `---SYSTEM`
- `---USER`
- `---ASSISTANT`

Rules:
- Each block runs until the next block header or end of body.
- If multiple `---SYSTEM` blocks exist, they are concatenated in order.
- The final system instruction used by the agent is:
  1) `instruction:` attribute (if any)
  2) + system instruction derived from body (prelude + `---SYSTEM` blocks)

### Interaction with multi-card parsing
- `---SYSTEM` / `---USER` / `---ASSISTANT` are **never** card boundaries.
- Only a valid frontmatter block (`---` + YAML with `type` + closing `---`) starts a new card.

---

## History Preload via `messages:`
- Two options:
  - inline history inside the file body (message blocks)
  - external history file(s) referenced via `messages` attribute
- If both are present, histories are merged in this order:
  1) external `messages` (in listed order)
  2) inline message blocks

### `messages` attribute shape
- `messages: ./history.md` (string)
- `messages: [./history.md, ./fewshot.json]` (list)

### Path resolution
- Relative paths in `messages` are resolved **relative to the card file directory**.

---

## MCP Servers and Tool Filters (YAML)
- Keep the shape as close to existing decorator semantics as possible.
- `servers` entries may reference `fastagent.config.yaml` by name or include inline server config
  using the same schema (transport, url, auth, etc.).
- Attributes not explicitly set on an inline server entry inherit from `fastagent.config.yaml`
  when a matching server name exists.
- `tools` filters allowed tools for that server (exact names or patterns). If omitted, inherits from
  `fastagent.config.yaml` or defaults to "all tools" for that server.

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

---

## Python References (`function_tools`, `tool_hooks`)
This RFC standardizes the reference format so the loader can resolve callables consistently.

### Reference string forms
- Module form: `package.module:callable`
- File form (relative): `./tools.py:callable` or `tools.py:callable`
- File form (absolute): `/abs/path/tools.py:callable` (discouraged; primarily for dev)

### Resolution
- If the left side contains a path separator (`/` or `\\`) or ends with `.py`, treat as a file path.
  - Relative file paths are resolved relative to the card file directory.
- Otherwise treat as a Python module import path.

### Error handling
- If a reference cannot be resolved or the callable is not callable, loader must fail with
  a message that includes the card name and the offending reference.

---

## Examples

### Agent with MCP servers, function tools, child agents, and hooks
Note: `---SYSTEM` is optional; if omitted, the body is treated as system instruction.
`---USER` and `---ASSISTANT` blocks are optional and seed message history.

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

### Multi-card markdown bundle
```md
---
type: agent
name: url_fetcher
servers:
  - fetch
---
Given a URL, provide a complete and comprehensive summary.

---
type: agent
name: social_media
---
Write a 280 character social media post for any given text.
Respond only with the post, never use hashtags.

---
type: chain
name: post_writer
sequence:
  - url_fetcher
  - social_media
instruction: "Generate a short social media post from a URL summary."
---
---USER
http://llmindset.co.uk
```

---

## Loading API
- `load_agents(path)` loads a file or a directory.
- Loading is immediate (no deferred mode).
- All loaded agents are tracked with a name and source file path.
- If a subsequent `load_agents(path)` call does not include a previously loaded agent from that
  path, the agent is removed.

## Hot Reload / Replace Behavior
- Internal representation keeps `file`/`url` origins and subscribes to updates by default.
- Provide CLI switch to choose reload mode:
  - `--reload-mode=hot-swap` (default): apply changes automatically on modify
  - `--reload-mode=explicit`: only reload via explicit command/tool call

## Tools Exposure (fast-agent-mcp)
Expose loader utilities via internal MCP tools, including update by name:
- `fast-agent-mcp.load_agents(path)`

---

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

---

## Appendix: AgentCard Samples
See `plan/agent-card-rfc-sample.md`.
