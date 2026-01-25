# VP ↔ evalstate (fast-agent) — подробный конспект переписки

Этот файл — сжатое, но детальное резюме переписки и контекста вокруг моих PR/идей для fast-agent, чтобы дальше готовить ответы/посты/следующие PR без потери деталей.

## Источники (где лежит первичка)

DM/личка:
- `/home/tools/report/fast-agent/evalstate-dm.md` — DM (19 Dec 2025 → 9 Jan 2026; частично).
- `/home/tools/report/fast-agent/ssmith4498_page_1.html` — DM export Discrub (27 Nov 2025 → 21 Jan 2026; более полный).

Discord server export (fast-agent-mcp):
- `/home/tools/report/fast-agent/fast-agent-mcp.zip` распакован в `/home/tools/report/fast-agent/fast-agent-mcp/`.
- Релевантные треды/страницы, где есть диалог `iqdoctor` ↔ `ssmith4498` (Display Name: `evalstate`):
  - `.../this looks very cool!_page_1.html`
  - `.../Good morning. My apologies if this is_page_1.html`
  - `.../There is a workaround if using @fast_page_1.html`
  - `.../ЁЯУг !fast-agent v0.4.16 is out!!_page_1.html`
  - `.../Heads up! \`fast-agent-mcp v0.4.19\` is_page_1.html`
  - `.../Release v0.4.22 ┬╖ evalstate!fast-agent_page_1.html`
  - `.../shell access_page_1.html`
  - `.../chain also does not retun a structured_page_1.html`
  - `.../for the brave, branch dev!0.4.35 has_page_1.html`
  - `.../Can you share the output of !fast-agent check! please_page_1.html`
  - `.../Poll notes_page_1.html`
  - `.../Feature! Agents-as-Tools тАФ expose child _page_1.html`
  - `.../general_page_1.html` (очень большой; в основном анонсы/контекст).

Доп. материалы (скорее как “черновики/контекст”):
- `/home/tools/report/fast-agent/fast-agent-issue-586-comment.md` — готовый комментарий для issue #586 (про skills disable).
- `/home/tools/report/fast-agent/fast-agent-v0.4.32-*.md`, `/home/tools/report/fast-agent/fast-agent-v0.4.37-discord-announcement.md` — наброски анонсов/конспекты релизов.
- `/home/tools/report/fast-agent/fast-agent.ai Authoritative Articles & Blog Posts.docx` — список статей/постов (контекст для docs-страницы “articles”).

## Участники

- Я: Valeriy Pavlovich (`iqdoctor`).
- Мейнтейнер: Shaun Smith (`evalstate` / `ssmith4498`).

## Большая картина (о чем в целом переписка)

1) Я активно контрибьючу в fast-agent (Agents-as-Tools, AgentCard, watch/reload, function_tools, quality fixes) и параллельно пытаюсь “упаковать” это в понятную историю: стандарты AgentCard, dev-loop через REPL, дистрибуция наборов карточек, и продвижение через LinkedIn/Discord.

2) evalstate двигает fast-agent в сторону:
- надежного tool-loop (в т.ч. hooks),
- ACP/observability (tool calls, /history, /mcp),
- компакции/истории (включая будущие стратегии),
- “self-modifying” workflows (LSP как skill, “skills for everything”),
- интеграции с Hugging Face/Toad + MCP server deployment,
- улучшения UX/CLI (shell access, reload/watch, sessions, и т.п.).

Ниже — подробности по темам/договоренностям и таймлайн.

## Темы и договоренности (по сути)

### 1) Agents-as-Tools: интеграция в main, семантика истории, ACP telemetry, auto-compact

Контекст:
- Я принёс Agents-as-Tools как ключевой workflow-паттерн (оркестратор вызывает дочерние агенты как tools).
- Мейнтейнер смотрел PR, добавлял свои коммиты (потому что не мог пушить в мой бранч), просил ретест на моей стороне.

Ключевые моменты из обсуждения:
- “Additive” фича: evalstate хотел убедиться, что изменения не ломают существующие потоки.
- История дочерних агентов:
  - дефолт: ребёнок клонирует history при старте и дропает её на завершении;
  - опционально: можно мерджить историю ребёнка обратно в orchestrator (и тут становится нужен auto-compact/стратегии компакции, чтобы история не разрасталась бесконтрольно).
- Telemetry/ACP:
  - evalstate прямо спрашивал, “эмитит ли это telemetry для ACP”.
- Planner:
  - обсуждали, что planner можно рассматривать как “массовый tool call” (LLM генерит JSON/plan → параллельный dispatch).
- UI/Web access:
  - я пытался связать идею web-chat UI со streaming + полными логами выполнения (для “business users”).

Артефакты/ссылки:
- PR/Issue: `fast-agent` PR #515, #552 (добавленные коммиты), docs PR #28 + PR #553 (align examples), issue #520.

### 2) “Function tools” vs MCP tools; ToolRunner; hooks

Вопрос/проблема:
- В fast-agent исторически “custom python tools” жили отдельным ToolAgent-паттерном, а хотелось “как в OpenAI Agents SDK”: любые python callables можно подключать как tools к обычному агенту, одновременно с MCP servers/tools и child agents.

Что обсуждали и к чему пришли:
- В Discord (тред про custom python tools) я отметил, что код явно рефакторится к единому “tool runner loop + hooks”, и PR #562 делает Function Tool pattern сильно проще, но “сам по себе” не даёт идеального UX.
- evalstate подтвердил, что проблема именно в “clean way to mix custom tools and MCP Servers on the standard McpAgent decorator” (внутри они это уже делают для human input/file/shell tools, но API надо раскрыть).

Отдельная ветка — “Hook Tool Declarative Spec”:
- Я предложил декларативный API на уровне `@fast.agent(...)` чтобы смешивать:
  - servers + MCP tool filters,
  - function_tools,
  - agents (agents-as-tools),
  - tool_hooks (до/после/вместо, mutate args/results, skip execution).
- evalstate ответил, что направление похоже на верное, но надо обсудить: breaking rename аргументов, загрузка из файла/формата (frontmatter/YAML vs JSON), совместимость с registry-форматами.

### 3) /card --tool и “не плодить отдельные code paths”

Проблема:
- /card --tool исторически превращал загруженную карточку в tool “особым способом”, что создаёт риск дублирования логики и “clutter” в списке tools.

Моя позиция/вывод (сформулирован в DM):
- В парадигме Agents-as-Tools любой загруженный агент потенциально может быть инструментом для другого агента, но “инструментами” должны становиться только те агенты, которые явным образом задекларированы (в orchestrator card).

Конкретное предложение, которое “сшивает” это:
- `/card --tool` НЕ должен напрямую инжектить tool.
- Он должен аппендить загруженного агента(ов) в `agents` текущего агента и “hot-swap” через уже существующий Agents-as-Tools flow.
- Это переиспользует 1 code path и убирает дублирование.

evalstate в ответ добавил “вектор”:
- мысль про “search”-парадигму: “find me a tool to do X” (меньше embedding-сложности, больше дешёвых LLM tokens для выбора инструмента).
- хотел бы иметь CLI UX: “load and switch” на skill/агента → использовать → discard.

Связанный issue:
- `fast-agent` issue #589 (brief: “find a tool/call a tool” парадигма).

### 4) --watch/--reload, server mode (ACP/MCP), hot reload AgentCards

Моя линия работ:
- PR #594: `--watch/--reload` в server mode (ACP/MCP), ACP `/reload`, MCP tool `reload_agent_cards`, docs updates.
- PR #597: “Fix exit traceback and watch tool files” (качество `--watch`).
- PR #603: “AgentCard --watch: minimal incremental refresh (mtime/size, per-card reload, safe parse)”.

Что важно по переписке:
- Я подчёркивал желание собрать полный “REPL development cycle” в прод-цепочке (ACP + mini_rag + watch + Telegram bot streaming).
- Мы обсуждали необходимость “environment/.fast-agent” как общей точки сборки (agents/tools/skills), и идею “agent card repo” как самостоятельного пакета (иногда даже без python-кода, только pyproject + зависимости + cards).

### 5) Skills: default vs disable/whitelist (issue #586, PR #588)

Суть бага/UX-проблемы:
- `skills=None` и `skills=[]` не давали очевидного пер-агентного “disable skills”, потому что:
  - FastAgent merges defaults когда `config.skills is None`,
  - McpAgent fallback’ится на `context.skill_registry` если `skill_manifests` пустой.

Обсуждение и решение-направление:
- Я предложил семантику:
  - `skills=None` → use defaults (как сейчас),
  - `skills=[]` → disable для этого агента.
- evalstate согласился, что “skills=None should disable loading for that agent maybe?”, и отметил CLI-опцию `--skills-dir` (и идею `--skills-dir none` как global-disable без размножения флагов).

Мой PR:
- PR #588: добавление `SKILLS_DEFAULT` sentinel (отличать “default behavior” от “explicit disable”), тесты/доки.

### 6) ACP tool titles: обрезка аргументов и прозрачность (PR #565)

Что случилось:
- Я заметил, что title для tool call в ACP формируется с `list(arguments.items())[:2]` + trim по длине, из-за чего title не отражает полный список аргументов (и создаёт ощущение “silent truncation”).

Позиции:
- evalstate: title — только для отображения; rawInput должен быть полным; title тримить нормально, иначе можно сломать UI огромными строками (например, если tool пишет файлы/контент).
- я: ок тримить по размеру, но важно явно показывать, что данные урезаны, чтобы не вводить в заблуждение при дебаге.

Результат:
- evalstate согласился, что `[:2]` “looks wrong”.
- Я открыл PR #565: “Include full tool args in ACP titles (trimmed to 50 chars)” + тесты.

### 7) RAG/mini-rag, примеры, и “быстрый путь в прод”

Я просил совет по “самому быстрому пути” к RAG стеку (Google Docs/Sheets/Docs, etc), потому что мы ранее часто грузили документы прямо в chatgpt.com/platform.openai.com (FileSearch).

Мой MVP-вектор:
- “самый компактный rag для google docs” как function tool:
  - `mini_rag(query, drive_id, top_k)`
  - авто-индексация на первом запросе.
- Хотел сделать это как пример в fast-agent (и далее как отдельный agentcard repo).

evalstate (философия/подход):
- “idea -> production” как step 1, потом оптимизация по реальным usage traces:
  1) превратить в детерминированный код с вылизанными tool descriptions,
  2) или in-context learning / fine-tune под usage.
- Интерес к “community server tool” на Hugging Face (и к идее маленьких моделей + тонкой настройки после сбора запросов).

### 8) Sessions/история: ACP vs long-lived боты, хранение контекста

Моя боль:
- В Telegram-ботах мне нужны persistent sessions (перезапуск без потери истории).
- ACP “в одном процессе” + stdio transport → рестарт клиента = потеря контекста, если не хранить явно.

Обсуждение:
- evalstate заинтересован в `agentclientprotocol.com` (RFD “session-list”) и в том, чтобы не плодить параллельные способы управления сессиями.
- Идея evalstate: сохранять `{sessionid}.json` каждый turn (я уточнил: скорее `{sessionid}.jsonl`).

### 9) Tool-loop reliability: “pending tool call in history” и deepseek context overflow

Отдельный класс проблем, которые всплывали:
- Ошибка “assistant message with tool_calls must be followed by tool messages…” (insufficient tool messages), особенно после длинных прогонов/ошибок/прерываний.
- DeepSeek max context overflow (131,072 tokens) при длинной сессии без полной компакции.

Реакция/советы evalstate:
- Обновиться до последних версий (на момент переписки: `uv tool install -U fast-agent-mcp` до 0.4.35).
- Использовать `/history fork` и `/history rewind` чтобы продолжить разговор после проблем.
- Для компакции — скорее “пакетировать пару tool hooks” с разными стратегиями.

Моё резюме в треде:
- В fast-agent нет “автоматической полной компакции всей истории по token budget”.
- Есть:
  - trim для tool-loop history (`trim_tool_history: true`),
  - ручные команды `/history ...`,
  - `use_history: false` для специальных агентов,
  - кастомные hooks, которые могут править `ctx.message_history` и перезагружать историю.

### 10) Shell access и structured chain

Shell access:
- evalstate радовался UX (Ctrl+Y/Ctrl+T), я описал как это “unlock” интерактивный dev workflow.

Structured chain:
- evalstate подтвердил: “structured path intentionally doesn’t call tools”.
- chain “needs a fix” чтобы structured step мог трансформировать/расширять output.
- Рекомендации: использовать `generate()` (возвращает PromptMessageExtended), забирать tool result напрямую, или делать это через tool hook; message_history можно программно “резать/править”.

## Таймлайн (с датами; ключевые вехи)

Ниже — не все сообщения подряд, а “опорные точки” по датам.

### 2025-11-24 … 2025-11-27 (вход и первые темы)
- (Discord) приветствие в fast-agent-mcp.
- DM: обсуждение ACP как “killer feature”; я дал доступ к приватному repo `strato-space/call`.
- evalstate: приоритеты — ACP/Toad/HF inference providers + поддержка MCP spec (2025-11-25).

### 2025-12-13 … 2025-12-14 (Agents-as-Tools: доведение до merge)
- Я ребейзнул Agents-as-Tools на актуальный 0.4.x, старался держать дифф минимальным.
- evalstate добавлял/правил коммиты поверх (PR #552), просил ретест (tests + production case).
- Обсуждали auto-compact при merge-back history, planners, telemetry для ACP.
- Итог: “great work, just merged”; просьба маленького апдейта на fast-agent.ai странице workflows.
- Docs: fast-agent-docs PR #28 “Agents As Tools overview” готов; PR #553 aligned examples; issue #520 “ready to close”.

### 2025-12-16 … 2025-12-17 (комьюнити + web UI + PR #562)
- Я спросил, можно ли постить announcement про Maker/Agents-as-Tools; ответ evalstate: “oh hell yes! make noise”.
- Я предложил vision “fast-agent.ai как web interface” с логами/стримингом/agent switching.
- evalstate: ACP закрывает большую часть потребности (кроме transport/side-channel), tool runner + telemetry сделают web UI проще.
- evalstate: фокус на compaction + skills deployment перед toad launch; observability (/history, /mcp) держит его в TUI.
- evalstate: PR #562 (function tools / API refactor), ссылка на MCP issue #1577.

### 2025-12-19 … 2025-12-20 (PR #565 и MCP spec links PR #1999)
- DM: я поймал `[:2]` в ACP tool titles → evalstate согласился “looks wrong” → PR #565.
- DM: я нашёл битые ссылки в MCP блоге → PR modelcontextprotocol #1999 (мержено).
- evalstate: планы — skill registry integration, потом tool loop + auto-compaction; предложения, чем помочь: sampling spec/tool loops, динамический attach/detach MCP servers, CIMD support.

### 2025-12-22 (Function tools вопрос в Discord)
- Я разобрал “tool runner loop refactor” + PR #562.
- evalstate: “key point — MCP tools vs just tools”, и “missing clean way to mix custom tools + MCP servers on decorator”.

### 2025-12-26 … 2025-12-27 (v0.4.16 + hooks spec)
- Я запостил highlights релиза v0.4.16.
- Обсуждение tool runner hooks и “custom agent type vs hooks для любых tools”.
- Я предложил “Hook Tool Declarative Spec” (mix servers/tools/function_tools/agents/hooks).
- evalstate: обсуждение про breaking rename + формат (frontmatter/YAML vs JSON).

### 2025-12-31 (AgentCard как “стандарт” + skills disable PR #588)
- Я писал про необходимость более активного комьюнити и желание сделать AgentCard стандартом.
- Тред про skills: семантика disable, идея whitelist; evalstate поддержал PR; я оформил PR #588 + перечислил issues, которые можно закрывать (уже есть merged PR).

### 2026-01-03 … 2026-01-09 (AgentCard, /card --tool, watch/reload, demo/video, sessions)
- DM/Discord: обсуждение “stateless tool calls” для subagent, `spawn_detached_instance`, перенос function tool descriptions в detached clones.
- Я сделал PR #592 (function_tools через @fast.agent → AgentConfig, tests, Vertex RAG пример).
- Я анонсировал следующий PR #594 (watch/reload в server mode), далее PR #597 и #603.
- Мы обсуждали /card --tool и пришли к варианту “append to agents list + reuse Agents-as-Tools flow”.
- evalstate: занимался toad integration, добавил `#` команду (send message to another agent → response into input buffer), готовит интеграцию с HF community server; заинтересован в sessions.
- Я дал YouTube demo (fast-agent-vertex-rag); evalstate хотел постить.
- Короткое обсуждение Claude Code CLI (скепсис evalstate).
- Sessions: идея `{sessionid}.jsonl` на каждый turn; интерес к ACP session-list RFD.
- Мелкий, но важный UX: если instruction не задан, лучше default instruction, а не error (evalstate спросил → я сказал “Yes!”).

### 2026-01-17 … 2026-01-21 (shell access, structured chain, codexplan oauth, deepseek overflow, “what’s next?”)
- Shell access: обсуждали UX и “agent can modify agents”.
- Structured chain: structured path не вызывает tools, но “needs fix”; совет делать трансформации через `generate()`/hooks.
- CodexPlan OAuth: обсуждение dev/0.4.35, залипания OAuth flow, идея reuse `~/.codex/auth.json`.
- DeepSeek overflow + tool_calls mismatch: совет обновиться + /history fork/rewind; обсуждение, что “full compaction” делается через hooks (а не built-in).
- “What’s next?”: evalstate пишет большой релиз-ноут с видео; фокус на self-modifying (LSP как skill, стратегии компакции per agent), плюс MCP integration (hot reload/detach) и observability для MCP server mode.

