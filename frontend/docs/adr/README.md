# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for the ComfyUI Frontend project.

## What is an ADR?

An Architecture Decision Record captures an important architectural decision made along with its context and consequences. ADRs help future developers understand why certain decisions were made and provide a historical record of the project's evolution.

## ADR Index

| ADR                                                 | Title                                    | Status   | Date       |
| --------------------------------------------------- | ---------------------------------------- | -------- | ---------- |
| [0001](0001-merge-litegraph-into-frontend.md)       | Merge LiteGraph.js into ComfyUI Frontend | Accepted | 2025-08-05 |
| [0002](0002-monorepo-conversion.md)                 | Restructure as a Monorepo                | Accepted | 2025-08-25 |
| [0003](0003-crdt-based-layout-system.md)            | Centralized Layout Management with CRDT  | Proposed | 2025-08-27 |
| [0004](0004-fork-primevue-ui-library.md)            | Fork PrimeVue UI Library                 | Rejected | 2025-08-27 |
| [0005](0005-remove-importmap-for-vue-extensions.md) | Remove Import Map for Vue Extensions     | Accepted | 2025-12-13 |

## Creating a New ADR

1. Copy the template below
2. Name it with the next number in sequence: `NNNN-descriptive-title.md`
3. Fill in all sections
4. Update this index
5. Submit as part of your PR

## ADR Template

```markdown
# N. Title

Date: YYYY-MM-DD

## Status

[Proposed | Accepted | Rejected | Deprecated | Superseded by [ADR-NNNN](NNNN-title.md)]

## Context

Describe the issue that motivated this decision and any context that influences or constrains the decision.

- What is the problem?
- Why does it need to be solved?
- What forces are at play (technical, business, team)?

## Decision

Describe the decision that was made and the key points that led to it.

- What are we going to do?
- How will we do it?
- What alternatives were considered?

## Consequences

### Positive

- What becomes easier or better?
- What opportunities does this create?

### Negative

- What becomes harder or worse?
- What risks are we accepting?
- What technical debt might we incur?

## Notes

Optional section for additional information, references, or clarifications.
```

## ADR Status Values

- **Proposed**: The decision is being discussed
- **Accepted**: The decision has been agreed upon
- **Rejected**: The decision was not accepted
- **Deprecated**: The decision is no longer relevant
- **Superseded**: The decision has been replaced by another ADR

## Further Reading

- [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) by Michael Nygard
- [Architecture Decision Records](https://adr.github.io/) - Collection of ADR resources
