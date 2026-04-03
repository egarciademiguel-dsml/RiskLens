# Project Workflow

## How Work Gets Done

1. Define what needs to be built (scope, constraints, acceptance criteria).
2. Create a ticket in `docs/tickets/`.
3. Implement exactly what the ticket specifies.
4. Verify it works (run it, test it).
5. Update docs and ticket registry.

## Ticket Structure

Each ticket (`docs/tickets/RL-XXX.md`) must include:

- **ID** and **Title**
- **Objective** — what it achieves
- **Scope** — what's included and excluded
- **Files changed**
- **Acceptance criteria**
- **Decisions** — key choices made and why

## Scope Control

- Work only on what the ticket specifies.
- Do not add files, dependencies, or logic beyond scope.
- If something is ambiguous, clarify before implementing.

## Definition of Done

- All acceptance criteria are met.
- Code runs without errors.
- Tests pass (where applicable).
- Docs are updated.
- Changes are committed: `[RL-XXX] short description`

## Bug Handling

- Document in `docs/bug_reports/` with description, reproduction steps, and affected files.
- Fix follows the same ticket workflow.

## Decision Tracking

Key decisions and assumptions are recorded in each ticket's "Decisions" section. For cross-cutting decisions, use `docs/decisions/`.
