# 2. Restructure ComfyUI_frontend as a monorepo

Date: 2025-08-25

## Status

Proposed

<!-- [Proposed | Accepted | Rejected | Deprecated | Superseded by [ADR-NNNN](NNNN-title.md)] -->

## Context

[Most of the context is in here](https://github.com/Comfy-Org/ComfyUI_frontend/issues/4661)

TL;DR: As we're merging more subprojects like litegraph, devtools, and soon a fork of PrimeVue,
a monorepo structure will help a lot with code sharing and organization.

For more information on Monorepos, check out [monorepo.tools](https://monorepo.tools/)

## Decision

- Swap out NPM for PNPM
- Add a workspace for the PrimeVue fork
- Move the frontend code into its own app workspace
- Longer term: Extract and reorganize common infrastructure to take advantage of the new monorepo tooling

### Tools proposed

[PNPM](https://pnpm.io/) and [PNPM workspaces](https://pnpm.io/workspaces)

For monorepo management, I'd probably go with [Nx](https://nx.dev/), but I could be conviced otherwise.
There's a [whole list here](https://monorepo.tools/#tools-review) if you're interested.

## Consequences

### Positive

- Adding new projects with shared dependencies becomes really easy
- Makes the process of forking and customizing projects more structured, if not strictly easier
- It _could_ speed up the build and development process (not guaranteed)
- It would let us cleanly organize and release packages like `comfyui-frontend-types`

### Negative

- Monorepos take some getting used to
- Reviews and code contribution management has to account for the different projects' situations and constraints

<!-- ## Notes

Optional section for additional information, references, or clarifications. -->
