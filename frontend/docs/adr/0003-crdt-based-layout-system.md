# 3. Centralized Layout Management with CRDT

Date: 2025-08-27

## Status

Proposed

## Context

ComfyUI's node graph editor currently suffers from fundamental architectural limitations around spatial data management that prevent us from achieving key product goals.

### Current Architecture Problems

The existing system allows each node to directly mutate its position within LiteGraph's canvas renderer. This creates several critical issues:

1. **Performance Bottlenecks**: UI updates require full graph traversals to detect position changes. Large workflows (100+ nodes) can create bottlenecks during interactions due to this O(n) polling approach.

2. **Position Conflicts**: Multiple systems (LiteGraph canvas, DOMwidgets.ts overlays) currently compete to control node positions. Future Vue widget overlays will compound this maintenance burden.

3. **No Collaboration Foundation**: Direct position mutations make concurrent editing impossible—there's no mechanism to merge conflicting position updates from multiple users.

4. **Renderer Lock-in**: Spatial data is tightly coupled to LiteGraph's canvas implementation, preventing alternative rendering approaches (WebGL, DOM, other libraries, hybrid approaches).

5. **Inefficient Change Detection**: While LiteGraph provides some events, many operations require polling via changeTracker.ts. The current undo/redo system performs expensive diffs on every interaction rather than using reactive push/pull signals, creating performance bottlenecks and blocking efficient animations and viewport culling.

   This represents a fundamental architectural limitation: diff-based systems scale O(n) with graph complexity (traverse entire structure to detect changes), while signal-based reactive systems scale O(1) with actual changes (data mutations automatically notify subscribers). Modern frameworks (Vue 3, Angular signals, SolidJS) have moved to reactive approaches for precisely this performance reason.

### Business Context

- Performance issues emerge with workflow complexity (100+ nodes)
- The AI workflow community increasingly expects collaborative features (similar to Figma, Miro)
- Accessibility requirements will necessitate DOM-based rendering options
- Technical debt compounds with each new spatial feature

This decision builds on [ADR-0001 (Merge LiteGraph)](0001-merge-litegraph-into-frontend.md), which enables the architectural restructuring proposed here.

## Decision

We will implement a centralized layout management system using CRDT (Conflict-free Replicated Data Types) with command pattern architecture to separate spatial data from rendering behavior.

### Centralized State Management Foundation

This solution applies proven centralized state management patterns:

- **Centralized Store**: All spatial data (position, size, bounds, transform) managed in a single CRDT-backed store
- **Command Interface**: All mutations flow through explicit commands rather than direct property access
- **Observer Pattern**: Independent systems (rendering, interaction, layout) subscribe to state changes
- **Domain Separation**: Layout logic completely separated from rendering and UI concerns

This provides single source of truth, predictable state updates, and natural system decoupling—solving our core architectural problems.

### Core Architecture

1. **Centralized Layout Store**: A Yjs CRDT maintains all spatial data in a single authoritative store:

   ```typescript
   // Instead of: node.position = {x, y}
   layoutStore.moveNode(nodeId, { x, y })
   ```

2. **Command Pattern**: All spatial mutations flow through explicit commands:

   ```
   User Input → Commands → Layout Store → Observer Notifications → Renderers
   ```

3. **Observer-Based Systems**: Multiple independent systems subscribe to layout changes:
   - **Rendering Systems**: LiteGraph canvas, WebGL, DOM accessibility renderers
   - **Interaction Systems**: Drag handlers, selection, hover states
   - **Layout Systems**: Auto-layout, alignment, distribution
   - **Animation Systems**: Smooth transitions, physics simulations

4. **Reactive Updates**: Store changes propagate through observers, eliminating polling and enabling efficient system coordination.

### Implementation Strategy

**Phase 1: Parallel System**

- Build CRDT layout store alongside existing system
- Layout store initially mirrors LiteGraph changes via observers
- Gradually migrate user interactions to use command interface
- Maintain full backward compatibility

**Phase 2: Inversion of Control**

- CRDT store becomes single source of truth
- LiteGraph receives position updates via reactive subscriptions
- Enable alternative renderers and advanced features

### Why Centralized State + CRDT?

This combination provides both architectural and technical benefits:

**Centralized State Benefits:**

- **Single Source of Truth**: All layout data managed in one place, eliminating conflicts
- **System Decoupling**: Rendering, interaction, and layout systems operate independently
- **Predictable Updates**: Clear data flow makes debugging and testing easier
- **Extensibility**: Easy to add new layout behaviors without modifying existing systems

**CRDT Benefits:**

- **Conflict Resolution**: Automatic merging eliminates position conflicts between systems
- **Collaboration-Ready**: Built-in support for multi-user editing
- **Eventual Consistency**: Guaranteed convergence to same state across all clients

**Yjs-Specific Benefits:**

- **Event-Driven**: Native observer pattern removes need for polling
- **Selective Updates**: Only changed nodes trigger system updates
- **Fine-Grained Changes**: Efficient delta synchronization

## Consequences

### Positive

- **Eliminates Polling**: Observer pattern removes O(n) graph traversals, improving performance
- **System Modularity**: Independent systems can be developed, tested, and optimized separately
- **Renderer Flexibility**: Easy to add WebGL, DOM accessibility, or hybrid rendering systems
- **Rich Interactions**: Command pattern enables robust undo/redo, macros, and interaction history
- **Collaboration-Ready**: CRDT foundation enables real-time multi-user editing
- **Conflict Resolution**: Eliminates position "snap-back" behavior between competing systems
- **Better Developer Experience**: Clear separation of concerns and predictable data flow patterns

### Negative

- **Learning Curve**: Team must understand CRDT concepts and centralized state management
- **Migration Complexity**: Gradual migration of existing direct property access requires careful coordination
- **Memory Overhead**: Yjs library (~30KB) plus operation history storage
- **CRDT Performance**: CRDTs have computational overhead compared to direct property access
- **Increased Abstraction**: Additional layer between user interactions and visual updates

### Risk Mitigations

- Provide comprehensive migration documentation and examples
- Build compatibility layer for gradual, low-risk migration
- Implement operation history pruning for long-running sessions
- Phase implementation to validate approach before full migration

## Notes

This centralized state + CRDT architecture follows patterns from modern collaborative applications:

**Centralized State Management**: Similar to Redux/Vuex patterns in complex web applications, but with CRDT backing for collaboration. This provides predictable state updates while enabling real-time multi-user features.

**CRDT in Collaboration**: Tools like Figma, Linear, and Notion use similar approaches for real-time collaboration, demonstrating the effectiveness of separating authoritative data from presentation logic.

**Future Capabilities**: This foundation enables advanced features that would be difficult with the current architecture:

- Macro recording and workflow automation
- Programmatic layout optimization and constraints
- API-driven workflow construction
- Multiple simultaneous renderers (canvas + accessibility DOM)
- Real-time collaborative editing
- Advanced spatial features (physics, animations, auto-layout)

The architecture provides immediate single-user benefits while creating infrastructure for collaborative and advanced spatial features.

## References

- [Yjs Documentation](https://docs.yjs.dev/)
- [CRDTs: The Hard Parts](https://martin.kleppmann.com/2020/07/06/crdt-hard-parts-hydra.html) by Martin Kleppmann
- [Figma's Multiplayer Technology](https://www.figma.com/blog/how-figmas-multiplayer-technology-works/)
