# 4. Fork PrimeVue UI Library

Date: 2025-08-27

## Status

Rejected

## Context

ComfyUI's frontend requires modifications to PrimeVue components that cannot be achieved through the library's customization APIs. Two specific technical incompatibilities have been identified with the transform-based canvas architecture:

**Screen Coordinate Hit-Testing Conflicts:**

- PrimeVue components use `getBoundingClientRect()` for screen coordinate calculations that don't account for CSS transforms
- The Slider component directly uses raw `pageX/pageY` coordinates ([lines 102-103](https://github.com/primefaces/primevue/blob/master/packages/primevue/src/slider/Slider.vue#L102-L103)) without transform-aware positioning
- This breaks interaction in transformed coordinate spaces where screen coordinates don't match logical element positions

**Virtual Canvas Scroll Interference:**

- LiteGraph's infinite canvas uses scroll coordinates semantically for graph navigation via the `DragAndScale` coordinate system
- PrimeVue overlay components automatically trigger `scrollIntoView` behavior which interferes with this virtual positioning
- This issue is documented in [PrimeVue discussion #4270](https://github.com/orgs/primefaces/discussions/4270) where the feature request was made to disable this behavior

**Historical Overlay Issues:**

- Previous z-index positioning conflicts required manual workarounds (commit `6d4eafb0`) where PrimeVue Dialog components needed `autoZIndex: false` and custom mask styling, later resolved by removing PrimeVue's automatic z-index management entirely

**Minimal Update Overhead:**

- Analysis of git history shows only 2 PrimeVue version updates in 2+ years, indicating that upstream sync overhead is negligible for this project

**Future Interaction System Requirements:**

- The ongoing canvas architecture evolution will require more granular control over component interaction and event handling as the transform-based system matures
- Predictable need for additional component modifications beyond current identified issues

## Decision

We will **NOT** fork PrimeVue. After evaluation, forking was determined to be unnecessarily complex and costly.

**Rationale for Rejection:**

- **Significant Implementation Complexity**: PrimeVue is structured as a monorepo ([primefaces/primevue](https://github.com/primefaces/primevue)) with significant code in a separate monorepo ([PrimeUIX](https://github.com/primefaces/primeuix)). Forking would require importing both repositories whole and selectively pruning or exempting components from our workspace tooling, adding substantial complexity.

- **Alternative Solutions Available**: The modifications we identified (e.g., scroll interference issues, coordinate system conflicts) have less costly solutions that don't require maintaining a full fork. For example, coordinate issues could be addressed through event interception and synthetic event creation with scaled values.

- **Maintenance Burden**: Ongoing maintenance and upgrades would be very painful, requiring manual conflict resolution and keeping pace with upstream changes across multiple repositories.

- **Limited Tooling Support**: There isn't adequate tooling that provides the granularity needed to cleanly manage a PrimeVue fork within our existing infrastructure.

## Consequences

### Alternative Approach

- **Use PrimeVue as External Dependency**: Continue using PrimeVue as a standard npm dependency
- **Targeted Workarounds**: Implement specific solutions for identified issues (coordinate system conflicts, scroll interference) without forking the entire library
- **Selective Component Replacement**: Use libraries like shadcn/ui to replace specific problematic PrimeVue components and adjust them to match our design system
- **Upstream Engagement**: Continue engaging with PrimeVue community for feature requests and bug reports
- **Maintain Flexibility**: Preserve ability to upgrade PrimeVue versions without fork maintenance overhead

## Notes

- Technical issues documented in the Context section remain valid concerns
- Solutions will be pursued through targeted fixes rather than wholesale forking
- Future re-evaluation possible if PrimeVue's architecture significantly changes or if alternative tooling becomes available
- This decision prioritizes maintainability and development velocity over maximum customization control
