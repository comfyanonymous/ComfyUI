# 1. Merge LiteGraph.js into ComfyUI Frontend

Date: 2025-08-05

## Status

Accepted

## Context

ComfyUI's frontend architecture currently depends on a forked version of litegraph.js maintained as a separate package (@comfyorg/litegraph). This separation has created several architectural and operational challenges:

**Architectural Issues:**

- The current split creates a distributed monolith where both packages handle rendering, user interactions, and data models without clear separation of responsibilities
- Both frontend and litegraph manipulate the same data structures, forcing tight coupling across the frontend's data model, views, and business logic
- The lack of clear boundaries prevents implementation of modern architectural patterns like MVC or event-sourcing

**Operational Issues:**

- ComfyUI is the only known user of the @comfyorg/litegraph fork
- Managing separate repositories significantly slows developer velocity due to coordination overhead
- Version mismatches between frontend and litegraph cause recurring issues
- No upstream contributions to consider (original litegraph.js is no longer maintained)

**Future Requirements:**
The following planned features are blocked by the current architecture:

- Multiplayer collaboration requiring CRDT-based state management
- Cloud-based backend support
- Alternative rendering backends
- Improved undo/redo system
- Clear API versioning and compatibility layers

## Decision

We will merge litegraph.js directly into the ComfyUI frontend repository using git subtree to preserve the complete commit history.

The merge will:

1. Move litegraph source to `src/lib/litegraph/`
2. Update all import paths from `@comfyorg/litegraph` to `@/lib/litegraph`
3. Remove the npm dependency on `@comfyorg/litegraph`
4. Preserve the full git history using subtree merge

This integration is the first step toward restructuring the application along clear Model-View-Controller boundaries, with state mutations going through a single CRDT-mediated access point.

## Consequences

### Positive

- **Enables architectural refactoring**: Direct integration allows restructuring along proper MVC boundaries
- **Unblocks new features**: Multiplayer, cloud features, and improved undo/redo can now be implemented
- **Faster development**: Eliminates overhead of coordinating changes across two tightly-coupled packages
- **Better developer experience**: No more version mismatch issues or cross-repository debugging
- **Simplified maintenance**: One less repository to maintain, release, and version

### Negative

- **Larger repository**: The frontend repository will increase in size
- **Loss of versioning**: No more semantic versioning for litegraph changes
- **Maintenance responsibility**: Must maintain litegraph code directly
- **Historical references**: Past commit messages may reference issues from the original litegraph repository

## Notes

- Git subtree was chosen over submodules to provide a cleaner developer experience
- The original litegraph repository will be archived after the merge
- Future litegraph improvements will be made directly in the frontend repository
