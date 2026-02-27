# 6. PrimitiveNode Copy/Paste Lifecycle

Date: 2026-02-22

## Status

Proposed

## Context

PrimitiveNode creates widgets dynamically on connection. When copied, the clone has no `this.widgets`, so `LGraphNode.serialize()` drops `widgets_values` from the clipboard data. This causes secondary widget values (e.g., `control_after_generate`) to be lost on paste. See [WIDGET_SERIALIZATION.md](../WIDGET_SERIALIZATION.md#primitiveno-and-copypaste) for the full mechanism.

Related: [#1757](https://github.com/Comfy-Org/ComfyUI_frontend/issues/1757), [#8938](https://github.com/Comfy-Org/ComfyUI_frontend/pull/8938)

## Options

### A. Minimal fix: override `serialize()` on PrimitiveNode

Override `serialize()` to fall back to `this.widgets_values` (set during `configure()`) when the base implementation omits it due to missing `this.widgets`.

- **Pro**: No change to connection lifecycle semantics. Lowest risk.
- **Pro**: Doesn't affect workflow save/load (which already works via `onAfterGraphConfigured`).
- **Con**: Doesn't address the deeper design issue — primitives are still empty on copy.

### B. Clone-configured-instance lifecycle

On copy, the primitive is a clone of the configured instance (with widgets intact). On disconnect or paste without connections, it returns to empty state.

- **Pro**: Copy→serialize captures `widgets_values` correctly. Matches OOP expectations.
- **Pro**: Secondary widget state survives round-trips without special-casing.
- **Con**: `input.widget[CONFIG]` allows extensions to make PrimitiveNode create a _different_ widget than the target. Widget config is derived at connection time, not stored, so cloning the configured state may not be faithful.
- **Con**: Deserialization ordering — `configure()` runs before links are restored. PrimitiveNode needs links to know what widgets to create. `onAfterGraphConfigured()` handles this for workflow load, but copy/paste uses a different code path.
- **Con**: Higher risk of regressions in extension compatibility.

### C. Projection model (like Subgraph widgets)

Primitives act as a synchronization mechanism — no own state, just a projection of the target widget's resolved value.

- **Pro**: Cleanest conceptual model. Eliminates state duplication.
- **Con**: Primitives can connect to multiple targets. Projection with multiple targets is ambiguous.
- **Con**: Major architectural change with broad impact.

## Decision

Pending. Option A is the most pragmatic first step. Option B can be revisited after Option A ships and stabilizes.
