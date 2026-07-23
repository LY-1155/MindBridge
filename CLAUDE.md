## Agent skills

### Issue tracker

Issues live as markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the default five-role vocabulary with no overrides. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Available skills

All skills live under `.agents/skills/`. Each skill has a `SKILL.md` with a `name` and `description` frontmatter field specifying its triggers.

| Skill | Description | When to use |
|-------|-------------|-------------|
| **diagnose** | Disciplined debug loop: reproduce → minimise → hypothesise → instrument → fix → regression-test | Bug reports, broken/error/failing behavior, performance regressions. User says "diagnose this", "debug this" |
| **grill-with-docs** | Grilling session challenging plan against domain model, sharpens terminology, updates CONTEXT.md/ADRs inline | Stress-test a plan against project language and documented decisions. User says "grill with docs" |
| **grill-me** | Relentless interview about plan/design until shared understanding, resolving each branch of decision tree | User wants to be grilled on their design. User says "grill me" |
| **improve-codebase-architecture** | Find deepening opportunities: turn shallow modules into deep ones for testability and AI-navigability | User wants to improve architecture, find refactoring opportunities, consolidate tightly-coupled modules |
| **prototype** | Throwaway prototype to answer a design question. Two branches: terminal app for logic/state, or multiple UI variations | User wants to prototype, sanity-check data model/state machine, mock up UI, explore design options. User says "prototype this" |
| **tdd** | Test-driven development with red-green-refactor loop. Integration-style tests through public interfaces, vertical tracer-bullet slices | User wants to build features/fix bugs with TDD, mentions "red-green-refactor", wants integration tests |
| **to-issues** | Break plan/spec/PRD into independently-grabbable issues using vertical tracer-bullet slices | User wants to convert plan into issues, create implementation tickets, break down work |
| **to-prd** | Synthesize current conversation context into a PRD and publish to issue tracker | User wants to create PRD from current context. Silent — does not interview, just synthesizes |
| **triage** | Move issues through state machine: needs-triage → needs-info / ready-for-agent / ready-for-human / wontfix | User wants to create/triage issues, review bugs/feature requests, prepare issues for AFK agent, manage issue workflow |
| **zoom-out** | Go up a layer of abstraction, produce map of relevant modules and callers using domain vocabulary | User is unfamiliar with code section or needs big-picture perspective. User says "zoom out" |
| **caveman** | Ultra-compressed communication mode, ~75% token reduction while keeping full technical accuracy | User says "caveman mode", "talk like caveman", "less tokens", "be brief" |
| **handoff** | Compact conversation into handoff document for next agent to pick up | User wants to hand off current work to another session/agent |
| **write-a-skill** | Create new agent skills with proper structure, progressive disclosure, and bundled resources | User wants to create, write, or build a new skill |

When a user request matches a skill's "When to use" triggers, read the corresponding `SKILL.md` from `.agents/skills/<skill-name>/SKILL.md` and follow its instructions.
