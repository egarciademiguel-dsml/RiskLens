# Project Workflow

## Team Roles

- **Stakeholder** — Defines product goals, priorities, and acceptance criteria. Approves scope and deliverables.
- **Architect** — Designs system structure, writes tickets, defines technical constraints, and reviews implementation.
- **Implementer** — Executes tickets, writes code, and follows scope rules strictly.

## Standard Workflow Cycle

1. Stakeholder defines a goal or requirement.
2. Architect translates the requirement into one or more scoped tickets.
3. Implementer executes the ticket exactly as specified.
4. Architect reviews the output against the acceptance criteria.
5. Stakeholder approves or requests changes.

## Ticket Structure

Each ticket must include:

- **ID** — Unique identifier (e.g., RL-001).
- **Title** — Short descriptive name.
- **Objective** — What the ticket achieves.
- **Includes** — Exhaustive list of allowed actions.
- **Excludes** — Explicitly forbidden actions.
- **Files created/modified** — Full list of affected files.
- **Acceptance criteria** — Conditions that must be met for completion.
- **Notes** — Additional context or constraints.

## Scope Control Rules

- Work only on files listed in the ticket.
- Do not add files, dependencies, or logic beyond what is specified.
- Do not refactor or improve code outside the ticket scope.
- If a task is ambiguous, stop and ask the Architect.

## Definition of Done

A ticket is considered done when:

- All listed files are created or modified as specified.
- All acceptance criteria are satisfied.
- No forbidden actions were taken.
- Changes are committed following the commit convention.

## Bug Handling Policy

- Bugs are documented in `docs/bug_reports/` with a clear description, reproduction steps, and affected files.
- Bug fixes follow the same ticket workflow: scoped, reviewed, and approved.
- A bug fix must not introduce unrelated changes.

## Commit Convention

Format: `[RL-XXX] short description`

- One commit per ticket unless otherwise specified.
- Commit message must reference the ticket ID.
- Keep commits atomic and focused on the ticket scope.
