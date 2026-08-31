# ComfyUI node graphs in ten minutes

Read this first if you have not converted a pack before. It exists because the
alternative is inferring the model from the pack's source, which is slow and
tends to produce a plausible but wrong mental picture.

## What the thing is

ComfyUI is a **node graph editor whose graph is a program**. The user wires
nodes together; the frontend compiles that wiring into a JSON payload; a Python
backend executes it and streams results back.

```
user edits graph  →  graphToPrompt()  →  POST /prompt  →  backend executes
       ▲                                                        │
       └────────────── node.onExecuted(message) ◀───────────────┘
```

Two artifacts come out of the same graph, and confusing them is the single most
common conversion error:

|                     | **Workflow**                                               | **Prompt**                 |
| ------------------- | ---------------------------------------------------------- | -------------------------- |
| Purpose             | what the user saves and reloads                            | what the backend runs      |
| Contains            | positions, colours, titles, collapsed state, widget values | node inputs and links only |
| Produced by         | `graph.serialize()`                                        | `graphToPrompt()`          |
| Frontend-only nodes | present                                                    | must be resolved away      |

**Both must come out byte-identical after a conversion.** That is the hard
constraint: a user's saved file and the job they queue cannot change.

## What a node actually is

A node exists in two halves.

**The backend half** is a Python class. It declares its inputs, outputs and
category, and the server sends that declaration to the frontend as a **node
definition** — `nodeData` in the old hook. This is a _description_, not a node.

**The frontend half** is a JavaScript class generated _from_ that definition,
registered by type name. Every node the user drops on the canvas is an instance
of it.

```
node definition (from backend)  →  generated class  →  instances on the canvas
   "KSampler", inputs, outputs        KSampler            node #7, node #12
```

So there are three distinct things, and packs act on all three:

- **the definition** — before any class exists
- **the class** — affects every instance of that type
- **the instance** — one node on one canvas

A conversion that moves code between these levels changes behaviour. Watch for
it: `nodeData.name` is a _definition_; `this` inside `onNodeCreated` is an
_instance_.

## The lifecycle

Everything a pack hooks hangs off this sequence:

1. **Definitions arrive** from the backend.
2. **`beforeRegisterNodeDef`** — for each definition, every extension gets a
   chance to modify the definition and patch the about-to-be-registered class.
   _This is where nearly half of all packs do their work._
3. **`registerNodeType`** puts the class in the registry under its type name.
4. **Instance created** — user drops a node, or a workflow loads one.
   `onNodeCreated` fires (before the node has an id or a graph).
5. **`onAdded`** — the node joins the graph. Now it has an id and is
   addressable. _This is where the published `onCreated` fires, deliberately._
6. **`onConfigure`** — only for nodes loaded from a saved workflow; restores
   widget values and any pack-specific state.
7. **`onExecuted(message)`** — backend produced output for this node.
8. **`onRemoved`** — node deleted.

## Inputs, outputs, widgets

- **Inputs / outputs** are sockets. Links connect an output to an input.
- **Widgets** are the controls drawn _on_ the node — a seed number, a sampler
  dropdown, a text box. A widget holds a value that becomes an input at
  execution time.

The wrinkle: a widget can be **promoted to a socket** so another node can drive
it. Historically the frontend faked this by setting `widget.type =
'converted-widget'` and stashing the old type — a hack that packs learned to
detect and imitate. It is now a real property. When you see
`'converted-widget'`, the pack is almost always trying to _hide_ a widget, not
change its kind.

`widgets_values` in a saved workflow is a **positional array** — index matters,
names are not stored. This is why widget order and count are part of the wire
format, and why removing a widget is not a cosmetic change.

## Why packs customise the frontend at all

Packs are not being gratuitous. There are a handful of recurring motives, and
recognising which one you are looking at usually tells you the replacement.
Counts are sites across the ~5,000-pack registry corpus; a pack often has
several motives at once.

| Motive                                                          | What it looks like                                   | Scale               |
| --------------------------------------------------------------- | ---------------------------------------------------- | ------------------- |
| **Show backend output on the node** — text, previews, progress  | patch `onExecuted`, create a display widget          | 497 packs           |
| **Set up per-instance state** — dynamic inputs, defaults, DOM   | patch `onNodeCreated`                                | 943 packs           |
| **Restore that state on load**                                  | patch `onConfigure`                                  | 429 packs           |
| **React to wiring** — add a slot when the last one fills        | patch `onConnectionsChange`                          | 223 packs           |
| **Draw on the node** — badges, overlays, custom controls        | patch `onDrawForeground`                             | 199 packs           |
| **Define frontend-only nodes** — reroutes, switches, note nodes | `registerCustomNodes`, `isVirtualNode`               | 86 packs            |
| **Change what gets saved or queued**                            | patch `serialize`, `serializeValue`, `graphToPrompt` | fewer, highest risk |

The last row is where conversions do damage, because it is the row that touches
the wire format.

## Why any of this needs converting

Almost all of the above is done by **monkey-patching the generated class's
prototype**:

```js
const original = nodeType.prototype.onExecuted
nodeType.prototype.onExecuted = function (message) {
  original?.apply(this, arguments) // ← if you forget this, you break other packs
  myBehaviour.call(this, message)
}
```

Three things are wrong with this, and they motivate the whole published API:

1. **It reaches into internals.** `nodeType.prototype`, `node.widgets`,
   `link.origin_id` are implementation, and they are being reshaped.
2. **It does not compose.** Whether your handler survives depends on every
   other pack remembering to call through. One that forgets silently disables
   yours, and load order decides who wins.
3. **It cannot be undone.** There is no unpatch, so nothing can be torn down.

The published API replaces each of these with something registered rather than
patched: handlers are additive, ordered, and individually removable, and the
entity classes stay closed.

## The mental model to convert with

> A pack **declares** what it wants for **which node types**, and **reacts** to
> a small set of lifecycle events using **handles** that expose behaviour but
> not internals.

Concretely, the shape of nearly every conversion is:

```js
// before: run for every type, filter, patch the prototype
app.registerExtension({
  name: 'x',
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== 'MyNode') return // ← the selector
    const orig = nodeType.prototype.onExecuted // ← the chaining
    nodeType.prototype.onExecuted = function (m) {
      orig?.apply(this, arguments)
      populate.call(this, m.text) // ← the actual behaviour
    }
  }
})

// after: the selector is declared, the chaining disappears, behaviour is unchanged
comfy.defs.extend('MyNode', (b) => {
  b.onExecuted((node, result) => populate(node, result.text))
})
```

The guard clause becomes the selector. The capture-and-chain boilerplate goes
away entirely. What is left is the behaviour, which you should be changing as
little as possible.

## Things that will mislead you

- **`this` is not always a node.** Inside a patched prototype method it is an
  instance; inside `beforeRegisterNodeDef` it is not.
- **`nodeData` is a definition, not a node.** It has no widgets and no id.
- **A widget named the same is not the same widget.** Packs remove and recreate
  readout widgets on every execution.
- **`app` is the whole application.** A pack reaching for `app.graph` usually
  wants its own node's graph, and often only needs the node.
- **Absence of a hook means nothing.** Plenty of packs put their logic in a
  module-level side effect that runs at import.
