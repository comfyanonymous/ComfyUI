# Converting `beforeRegisterNodeDef` and prototype patching

**1,265 packs / 50.4M downloads / 47.4% of registry installs** register a
`beforeRegisterNodeDef` hook, and **1,191 of them use it to patch the generated
class's prototype**. This is the largest surface in the ecosystem — bigger than
the widget and painting cohorts combined — and the one that couples packs to our
internals most tightly.

## What they patch — measured

Prototype assignments across those 1,265 packs, litegraph methods only
(bundled-library noise like `_next`, `dispose`, `toString` filtered out):

| Patched method        | Sites | Packs   | Destination                    |
| --------------------- | ----- | ------- | ------------------------------ |
| `onNodeCreated`       | 3,161 | **943** | `b.onCreated(cb)`              |
| `onExecuted`          | 964   | **497** | `b.onExecuted(cb)`             |
| `onConfigure`         | 1,272 | **429** | `b.onConfigured(cb)`           |
| `onConnectionsChange` | 454   | 223     | `b.onConnectionsChanged(cb)`   |
| `onDrawForeground`    | 446   | 199     | `references/draw-callbacks.md` |
| `onRemoved`           | 353   | 158     | `b.onRemoved(cb)`              |
| `constructor`         | 473   | 49      | `b.onCreated(cb)`              |
| `getDefaultShape`     | 335   | 3       | no replacement — escalate      |

## The selector is already written — it's the guard clause

Every one of these hooks runs for **every registered node type**, so essentially
all of them open with a filter and return. That filter _is_ the selector:

```js
// before — ComfyUI-KJNodes jsnodes.js:37
async beforeRegisterNodeDef(nodeType, nodeData, app) {
  if (!nodeData?.category?.startsWith('KJNodes')) return
  switch (nodeData.name) {
    case 'ImageBatchMulti':
    case 'ImageAddMulti':
      nodeType.prototype.onNodeCreated = function () {
        setupDynamicInputs(this, { type: 'IMAGE', prefix: 'image_' })
      }
      break
    ...
  }
}

// after
comfy.defs.extend(['ImageBatchMulti', 'ImageAddMulti'], (b) => {
  b.onCreated((node) => setupDynamicInputs(node, { type: 'IMAGE', prefix: 'image_' }))
})
```

This is mechanical: lift the `if`/`switch` condition into the selector, one
`extend` call per case group.

It is also a real performance fix, not just tidiness. With 1,265 packs hooking
and a few thousand node types, boot currently runs **millions of callbacks that
immediately return**. A declarative predicate can be indexed.

Selector forms: exact type, array of types, `RegExp` over the type, or
`{ category }` where category is a string or a `RegExp`. The regex form covers
the prefix filter 53 packs open with —
`nodeData.category.startsWith('KJNodes')` becomes `{ category: /^KJNodes/ }`.

`onCreated` fires when the node **joins a graph**, not inside the constructor.
Litegraph's `onNodeCreated` runs before the node has an id, a graph, or store
registration, so widget writes made there are lost on insert. If a pack relied
on running before insertion, escalate — that ordering is not reproducible.

## Chaining boilerplate disappears

```js
// before — capture-and-chain, because prototype patching has no composition
const onExecuted = nodeType.prototype.onExecuted
nodeType.prototype.onExecuted = function (message) {
  onExecuted?.apply(this, arguments)
  populate.call(this, message.text)
}

// after — registered callbacks compose by construction
b.onExecuted((node, result) => populate(node, result.text))
```

Whether the old form worked at all depended on load order and on every pack
remembering to call through. Two packs patching the same method with one
forgetting silently broke the other.

## `onExecuted` — the second-largest, and easy to get wrong

497 packs. The result shape is now explicit rather than a raw backend payload:

```js
// before
nodeType.prototype.onExecuted = function (message) {
  populate.call(this, message.text)
}

// after
b.onExecuted((node, result) => populate(node, result.text))
```

`ExecutionResult` exposes `images`, `text`, and `raw` for everything else.
Custom output keys survive in `raw` — ADR 0007's passthrough schema guarantees
it — so a pack reading a bespoke key keeps working.

## A worked example touching four surfaces at once

ComfyUI-Custom-Scripts `showText.js:10` is representative of the harder cases:

```js
// before
async beforeRegisterNodeDef(nodeType, nodeData, app) {
  if (nodeData.name === 'ShowText|pysssss') {
    function populate(text) {
      if (this.widgets) {
        // On older frontend versions there is a hidden converted-widget
        const isConvertedWidget = +!!this.inputs?.[0].widget
        for (let i = isConvertedWidget; i < this.widgets.length; i++) {
          this.widgets[i].onRemove?.()
        }
        this.widgets.length = isConvertedWidget
      }
      for (const l of text) {
        const w = ComfyWidgets.STRING(this, 'text_' + this.widgets?.length, ...).widget
        w.inputEl.readOnly = true
        w.inputEl.style.opacity = 0.6
      }
    }
    ...
  }
}
```

Four separate conversions:

| Old                                                    | New                         |
| ------------------------------------------------------ | --------------------------- |
| `nodeData.name === 'ShowText\|pysssss'` guard          | the `defs.extend` selector  |
| `+!!this.inputs?.[0].widget` converted-widget sniffing | `input.isWidgetInput`       |
| `this.widgets.length = n` truncation                   | remove by name (below)      |
| `w.inputEl` DOM access                                 | `widgets.mount({ render })` |

Truncation has no single-call replacement, deliberately — assigning `length`
skips each widget's teardown, which is why the pack has to call `onRemove()` by
hand first:

```js
// after — removal runs teardown for you
for (const name of node.widgets.names().slice(keep)) {
  node.widgets.remove(name)
}
```

Note `isConvertedWidget` exists only to skip a widget the _old frontend_ hid via
the converted-widget protocol. With `setHidden()` as a real property, that
whole line of reasoning goes away — check `input.isWidgetInput` if you actually
care whether an input is a widget's socket form.

## Pack-owned node types — `registerCustomNodes` / `extends LGraphNode`

86 packs / 18.2% of installs define their own types by subclassing. The
replacement is `comfy.defs.define(...)`: the definition is plain data, and no
class is ever yours.

**First identify the intent.** "Virtual node" is four different things, and the
conversion differs for each:

| The node exists to…                     | Examples                     | Convert to                                                  |
| --------------------------------------- | ---------------------------- | ----------------------------------------------------------- |
| annotate — never executes               | Note nodes                   | `execution: 'frontend'`, no `resolve`                       |
| be a wire — indirection                 | Reroute, `SetNode`/`GetNode` | `resolve` returning `forwardTo`                             |
| be a value — UI-held literal            | Primitive, constants         | `resolve` returning `literal`                               |
| act on _other_ nodes — a remote control | rgthree Fast Muter           | **not `resolve` at all** — handle calls in widget callbacks |

```js
// before
class GetNode extends LGraphNode {
  constructor() {
    super()
    this.addOutput('value', '*')
    this.isVirtualNode = true
  }
  applyToGraph() {
    /* rewrites links on the LIVE graph mid-serialize */
  }
}
LiteGraph.registerNodeType('GetNode', GetNode)

// after — declared, and resolution is a pure answer over a read-only view
comfy.defs.define({
  type: 'GetNode',
  execution: 'frontend',
  outputs: [{ name: 'value', type: '*' }],
  widgets: [{ type: 'string', name: 'key', value: '' }],
  resolve: ({ self, nodesOfType }) => {
    const setter = nodesOfType('SetNode').find(
      (n) => n.widgetValue('key') === self.widgetValue('key')
    )
    return {
      value: setter ? { forwardTo: setter.input('value') } : { omit: true }
    }
  }
})
```

`resolve` answers, per output: `{ forwardTo: inputRef }` ("whatever feeds that
input"), `{ literal: value }`, or `{ omit: true }`. Our pass follows chains
(Get → Set → Reroute → …) with cycle detection. You never see or touch the
prompt being built — a resolver that throws poisons one prompt build and the
graph is untouched, which is exactly what `applyToGraph` could not guarantee.
A simple reroute is one line: `resolve: ({ self }) => ({ out: { forwardTo:
self.input('in') } })`.

**Nodes that act on other nodes are ordinary nodes.** A Fast Muter is a remote control: a row of toggles, one per wired-in node. A Fast Muter is `defs.define` with
`execution: 'frontend'`, buttons via `widgets`, and callbacks that call
`comfy.graph.nodesOfType(...)` + `node.setMode('bypass')` — edit-time changes
the user can undo, not serialization behaviour. Do not put neighbour mutation
in `resolve`; `resolve` cannot write anything, by design.

**Do not carry over** `isVirtualNode`, `applyToGraph`, or the subclass itself.
If the node's `applyToGraph` does something none of the three resolution shapes
can express, that is an `api-gap` punt — name what it rewrites.

## Node handles are accessors, not properties

Every read and write on a `NodeHandle` is a method — the contract in
`src/types/extensionV2.ts`, so a read can be a store query and a write can
become a command. Property syntax silently does nothing or throws.

| Old (on the raw node)    | Published API                               |
| ------------------------ | ------------------------------------------- |
| `node.title = t`         | `node.setTitle(t)` / `getTitle()`           |
| `node.color = c`         | `node.setColor(c)` / `getColor()`           |
| `node.bgcolor = c`       | `node.setBgColor(c)` / `getBgColor()`       |
| `node.mode = 4`          | `node.setMode('bypass')` / `getMode()`      |
| `node.flags.collapsed`   | `node.isCollapsed()` / `setCollapsed(b)`    |
| `node.flags.pinned`      | `node.isPinned()` / `setPinned(b)`          |
| `node.shape = s`         | `node.setShape(s)` / `getShape()`           |
| `node.properties[k] = v` | `node.setProperty(k, v)` / `getProperty(k)` |
| `node.size` / `node.pos` | `node.getSize()` / `getPosition()`          |

**Both setters take one tuple**, not two numbers — the legacy `setSize(w, h)`
and `setPos(x, y)` arities are gone:

```js
// before
node.setSize([200, 58]) // litegraph took an array here
node.color = '#1b4669'
node.bgcolor = '#29699c'

// after
node.setSize([200, 58]) // unchanged — still one tuple
node.setColor('#1b4669')
node.setBgColor('#29699c')
```

Widget handles follow the same rule: `getValue()`/`setValue()`,
`isHidden()`/`setHidden()`, `getOptions()`/`setOption()`, and `widgetType`
rather than `type`. See `widgets.md`.

## Per-instance state

Handles hold no arbitrary properties, so `node._myState = x` has no target. Keep
the state yourself, keyed by node id:

```js
// before — a property stashed on the node instance
nodeType.prototype.onExecuted = function (output) {
  if (this._lastHash === hash) return
  this._lastHash = hash
}

// after — the pack owns its own state
const stateByNode = new Map()
const stateFor = (id) => {
  let s = stateByNode.get(id)
  if (!s) stateByNode.set(id, (s = {}))
  return s
}

comfy.defs.extend('MyNode', (b) => {
  b.onExecuted((node, result) => {
    const state = stateFor(node.id)
    if (state.lastHash === hash) return
    state.lastHash = hash
  })
  b.onRemoved((node) => stateByNode.delete(node.id))
})
```

This is a supported conversion, not a workaround. Clean up in `onRemoved` — the
old form was collected with the node, and a Map is not.

## Traps

**The constructor self-assignment.** `this.type = this.type ?? undefined` is a
defensive no-op that now throws, killing construction entirely. Handled
mechanically by the `type-write-noop` rule — 8 packs, 3.86M downloads, including
rgthree's whole 12-type virtual-node family.

**`nodeData` is a definition, not a node.** Reading `nodeData.input.required.x`
is fine; it becomes `b.def.inputs`. But the shape differs — do not assume a
field-by-field rename.

**Async hooks.** `async beforeRegisterNodeDef` is common and usually gratuitous.
`defs.extend` is synchronous; if a pack genuinely needs async setup, do it in
`onCreated` and guard for the node being removed before it resolves.

**Order-dependent patches.** A pack that reads `nodeType.prototype.onNodeCreated`
expecting another pack's patch to already be installed has no equivalent, by
design. Escalate — it needs the author.

## Source data

Prototype-assignment counts derived from the corpus at
`~/comfy/nodes-compat-study/corpus/registry_js` (4,969 packs, 2,562 files
scanned). Grep-derived; sample-verify before citing an individual number.
