# API

This document is intended to provide a brief introduction to new Litegraph APIs.

<detail open>

<summary>

# CanvasPointer API

</summary>

CanvasPointer replaces much of the original pointer handling code. It provides a standard click, double-click, and drag UX for users.

<detail open>

<summary>

## Default behaviour changes

</summary>

- Dragging multiple items no longer requires that the shift key be held down
  - Clicking an item when multiple nodes / etc are selected will still deselect everything else
- Clicking a connected link on an input no longer disconnects and reconnects it
- Double-clicking requires that both clicks occur nearby
- Provides much room for extension, configuration, and changes

#### Bug fixes

- Intermittent issue where clicking a node slightly displaces it
- Alt-clicking to add a reroute creates two undo steps

### Selecting multiple items

- `Ctrl + drag` - Begin multi-select
- `Ctrl + Shift + drag` - Add to selection
  - `Ctrl + drag`, `Shift` - Alternate add to selection
- `Ctrl + drag`, `Alt` - Remove from selection

### Click "drift"

A small amount of buffering is performed between down/up events to prevent accidental micro-drag events. If either of the two controls are exceeded, the event will be considered a drag event, not a click.

- `buffterTime` is the maximum time that tiny movements can be ignored (Default: 150ms)
- `maxClickDrift` controls how far a click can drift from its down event before it is considered a drag (Default: 6)

### Double-click

When double clicking, the double click callback is executed shortly after one normal click callback (if present). At present, dragging from the second click simply invalidates the event - nothing will happen.

- `doubleClickTime` is the maximum time between two `down` events for them to be considered a double click (Default: 300ms)
- Distance between the two events must be less than `3 * maxClickDrift`

### Configuration

All above configuration is via class static.

```ts
CanvasPointer.bufferTime = 150
CanvasPointer.maxClickDrift = 6
CanvasPointer.doubleClickTime = 300
```

</detail>

<detail open>

<summary>

## Implementing

</summary>

Clicking, double-clicking, and dragging can now all be configured during the initial `pointerdown` event, and the correct callback(s) will be executed.

A click event can be as simple as:

```ts
if (node.isClickedInSpot(e.canvasX, e.canvasY))
  this.pointer.onClick = () => node.gotClickInSpot()
```

Full usage can be seen in the old `processMouseDown` handler, which is still in place (several monkey patches in the wild).

### Registering a click or drag event

Example usage:

```typescript
const { pointer } = this
// Click / double click - executed on pointerup
pointer.onClick = (e) => node.executeClick(e)
pointer.onDoubleClick = node.gotDoubleClick

// Drag events - executed on pointermove
pointer.onDragStart = (e) => {
  node.isBeingDragged = true
  canvas.startedDragging(e)
}
pointer.onDrag = () => {}
// finally() is preferred where possible, as it is guaranteed to run
pointer.onDragEnd = () => {}

// Always run, regardless of outcome
pointer.finally = () => (node.isBeingDragged = false)
```

## Widgets

Adds `onPointerDown` callback to node widgets. A few benefits of the new API:

- Simplified usage
- Exposes callbacks like "double click", removing the need to time / measure multiple pointer events
- Unified UX - same API as used in the rest of Litegraph
- Honours the user's click speed and pointer accuracy settings

#### Usage

```ts
// Callbacks for each pointer action can be configured ahead of time
widget.onPointerDown = function (pointer, node, canvas) {
  const e = pointer.eDown
  const offsetFromNode = [e.canvasX - node.pos[0], e.canvasY - node.pos[1]]

  // Click events - no overlap with drag events
  pointer.onClick = (upEvent) => {
    // Provides access to the whole lifecycle of events in every callback
    console.log(pointer.eDown)
    console.log(pointer.eMove ?? "Pointer didn't move")
    console.log(pointer.eUp)
  }
  pointer.onDoubleClick = (upEvent) => this.customFunction(upEvent)

  // Runs once before the first onDrag event
  pointer.onDragStart = () => {}
  // Receives every movement event
  pointer.onDrag = (moveEvent) => {}
  // The pointerup event of a drag
  pointer.onDragEnd = (upEvent) => {}

  // Semantics of a "finally" block (try/catch).  Once set, the block always executes.
  pointer.finally = () => {}

  // Return true to cancel regular Litegraph handling of this click / drag
  return true
}
```

</detail>

### TypeScript & JSDoc

In-IDE typing is available for use in at least mainstream editors. TypeScript definitions are available in the litegraph library.

```ts
/** @import { IWidget } from './path/to/litegraph/litegraph.d.ts' */
/** @type IWidget */
const widget = node.widgets[0]
widget.onPointerDown = function (pointer, node, canvas) {}
```

#### VS Code

![image](https://github.com/user-attachments/assets/e14afd02-247f-44dc-acbf-6333027cd488)

## Hovering over

Adds API for downstream consumers to handle custom cursors. A bitwise enum of items,

```typescript
type LGraphCanvasState = {
  /** If `true`, pointer move events will set the canvas cursor style. */
  shouldSetCursor: boolean,
  /** Bit flags indicating what is currently below the pointer. */
  hoveringOver: CanvasItem,
  ...
}

// Disable litegraph cursors
canvas.state.shouldSetCursor = false

// Checking state - bit operators
if (canvas.state.hoveringOver & CanvasItem.ResizeSe) element.style.cursor = 'se-resize'
```

</detail>

# Removed public interfaces

All are unused and incomplete. Have bugs beyond just typescript typing, and are (currently) not worth maintaining. If any of these features are desired down the track, they can be reimplemented.

- Live mode
- `dragged_node`

## LiteGraph

These features have not been maintained, and would require refactoring / rewrites. As code search revealed them to be unused, they are being removed.

- addNodeMethod
- compareObjects
- auto_sort_node_types (option)
