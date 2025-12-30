# AgentCard Multi-card Bundles (Experimental)

This document defines the optional/experimental multi-card format. The default and
recommended path remains **one card per file**.

## Scope
- Multi-card files allow bundling several AgentCards into a single `.md` or `.yaml`.
- This is optional/experimental and may change; loaders should treat it as opt-in.

## YAML Bundle (multi-doc YAML stream)
A YAML file may contain multiple YAML documents separated by `---` markers. Each
document is one AgentCard.

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

## Markdown Bundle (multi-card `.md`)
A markdown bundle contains repeated frontmatter blocks. Each block defines one card.

Example:
```md
---
type: agent
name: url_fetcher
---
Given a URL, provide a complete and comprehensive summary.

---
type: agent
name: social_media
---
Write a 280 character social media post for any given text.
Respond only with the post, never use hashtags.
```

## Card Boundary and Parsing Rules (Markdown)

### Frontmatter definition
- Frontmatter starts at a line that is exactly `---` (column 0, no leading/trailing spaces).
- Frontmatter ends at the next line that is exactly `---`.
- The text between them is parsed as YAML.

### What counts as a new card?
Not every `---` in markdown is a card boundary. The loader uses this rule:

> A `---` line starts a new card only if the subsequent YAML parses successfully and
> contains a `type` key.

Operationally:
1. Scan for a `---` line.
2. Parse YAML until the next closing `---`.
3. If YAML contains `type`, treat it as a card start.
4. Otherwise treat it as normal markdown content and continue scanning.

### Body range
- The card body begins immediately after the closing frontmatter delimiter.
- The card body ends at the start of the next valid card or end-of-file.

### Name rules
- In multi-card files, `name` is required on every card.
- Loader must reject duplicate `name`s within the load-set.

## Interaction with history files
- Inline `---USER` / `---ASSISTANT` blocks are not supported in AgentCard bodies.
- Use external `messages` files for history (see main RFC).
