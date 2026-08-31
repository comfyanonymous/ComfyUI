# Converting widget-array mutation and the converted-widget protocol

> **Accessor style.** The published handles follow `src/types/extensionV2.ts`
> (PR #11251): reads and writes are methods, not properties —
> `widget.getValue()` / `setValue(v)`, `isHidden()` / `setHidden(b)`,
> `getOptions()` / `setOption(key, value)`, `setLabel(s)`, and `widgetType` for
> the type. Node handles keep `getTitle()`/`setTitle()` style likewise.

The largest cohort by pack count, and the one where naive conversions are most
likely to be silently wrong.

| Surface                   | Packs | Installs |
| ------------------------- | ----- | -------- |
| `widgets.splice`          | 286   | 21.6%    |
| `widget.type` overwrite   | 270   | 18.6%    |
| converted-widget protocol | 238   | 25.1%    |
| `widgets = [...]`         | 226   | 10.9%    |
| `widgets.push`            | 142   | —        |
| `getCustomWidgets` POJO   | 91    | 16.8%    |
| `widgets.length = 0`      | 31    | —        |

These overlap heavily; the same pack usually hits several.

## Classify before converting — `splice` is often not a reorder

```js
// ComfyUI-KJNodes setgetnodes.js:988
// Fresh options object (live getter preserved) + remove/re-add to force
// Vue re-extraction.
w.options = newOpts
const idx = this.widgets.indexOf(w)
if (idx >= 0) {
  this.widgets.splice(idx, 1)
  this.widgets.splice(idx, 0, w)
}
```

Removed and reinserted **at the same index** — the array is unchanged. This is
cache invalidation, not reordering. It converts to:

```js
node.widgets.get(name).setOption('values', newValues)
```

The hack disappears: invalidation is the API's problem now. A converter that
assumes "splice means reorder" produces nonsense here.

**Read the indices before choosing a rule.** Same index in and out → invalidation.
Different index → a real move (`widgets.move`). A full rewrite of the array →
`widgets.reorder`.

## The converted-widget protocol — ~20 lines become one property

The full pattern, from ComfyUI-Easy-Use `easyExtraMenu.js:339`:

```js
const CONVERTED_TYPE = 'converted-widget'

function hideWidget(node, widget, suffix = '') {
  widget.origType = widget.type
  widget.origComputeSize = widget.computeSize
  widget.origSerializeValue = widget.serializeValue
  widget.computeSize = () => [0, -4] // -4 offsets litegraph's inter-widget gap
  widget.type = CONVERTED_TYPE + suffix
  widget.serializeValue = () => {
    if (!node.inputs) return undefined
    const input = node.inputs.find((i) => i.widget?.name === widget.name)
    if (!input || !input.link) return undefined // unlinked → do not serialize
    return widget.origSerializeValue
      ? widget.origSerializeValue()
      : widget.value
  }
  if (widget.linkedWidgets) {
    for (const w of widget.linkedWidgets) hideWidget(node, w, ':' + widget.name)
  }
}
```

Everything above exists to emulate a property that did not exist:

```js
node.widgets.get(name).hidden = true
```

Gone with it: the `origType`/`origComputeSize`/`origSerializeValue` save-and-
restore dance, the `[0, -4]` magic number, and the type-string mangling.

### ⚠️ The trap: `hidden` does not imply "do not serialize"

The old hack **coupled** two things. Hiding a widget also installed a
`serializeValue` that returned `undefined` unless the matching input was linked.

In the published API those are orthogonal — `hidden` is presentation,
`serialize` is persistence. That is the better design, but it means a literal
one-line conversion **changes behaviour**: a hidden widget now serializes where
it previously did not.

So check what the pack relied on:

- Hiding purely for presentation → `hidden = true` is complete.
- Hiding _and_ suppressing serialization → set `hidden`, and handle
  serialization explicitly. If the value should persist only when the socket is
  connected, that condition belongs in the pack's own `onSerialize`.

This is a wire-format change if you get it wrong, so it will be caught by the
gate — but understand it rather than letting the gate find it.

### Linked widgets

The recursion over `widget.linkedWidgets` (seed + seed-control being the classic
pair) has **no published equivalent**. Hide each widget explicitly by name, or
escalate if the linkage is computed rather than fixed.

## `widget.type` overwrite — 270 packs

Almost always the converted-widget hack above. If it is genuinely trying to
change a widget's _kind_, there is no replacement: type is identity. Remove the
widget and add the intended one.

```js
// before
widget.type = 'converted-widget' // → widget.setHidden(true)

// before — genuinely changing kind
widget.type = 'combo' // → remove + add, or escalate
```

## Array assignment and truncation

```js
// before
this.widgets = this.widgets.filter((w) => w.name !== 'seed')
this.widgets.length = 0
this.widgets.push(w)

// after
node.widgets.remove('seed')
for (const name of node.widgets.names()) node.widgets.remove(name)
node.widgets.add(def)
```

Two reasons not to translate these literally:

- **Assigning a new array drops the renderer's tracking.** The array identity is
  what the renderer watches; `widgets.reorder` splices in place for exactly this
  reason.
- **Assigning `length` skips teardown.** Packs that do it correctly call
  `widget.onRemove?.()` first — see Custom-Scripts `showText.js`. `remove()`
  runs teardown for you, so the manual loop goes away.

## Creating a widget

`ComfyWidgets.*` is an unpublished internal. `node.widgets.add(def)` replaces it,
and the `def` is plain data — no `node`, no `app`, no return-value unwrapping:

```js
// before
const w = ComfyWidgets.STRING(
  this,
  'text',
  ['STRING', { multiline: true }],
  app
).widget
w.inputEl.readOnly = true
w.inputEl.style.opacity = 0.6

// after
const widget = node.widgets.add({
  type: 'textarea',
  name: 'text',
  value: '',
  disabled: true
})
```

`disabled: true` replaces the `readOnly` + `opacity` pair — the two lines packs
use to fake a read-only widget. It is a real property, so the styling stays
consistent with every other disabled widget instead of being hand-rolled.

| Old factory                                     | `type`                                      |
| ----------------------------------------------- | ------------------------------------------- |
| `ComfyWidgets.STRING(..., { multiline: true })` | `'textarea'`                                |
| `ComfyWidgets.STRING(...)`                      | `'string'`                                  |
| `ComfyWidgets.INT` / `FLOAT`                    | `'number'` (or `'slider'` with `min`/`max`) |
| `ComfyWidgets.BOOLEAN`                          | `'toggle'`                                  |
| `ComfyWidgets.COMBO`                            | `'combo'`, values via `options.values`      |
| `ComfyWidgets.MARKDOWN`                         | `'markdown'`                                |
| `ComfyWidgets.COLOR`                            | `'color'`                                   |

### Keeping a widget out of the saved workflow

```js
// before
widget.serializeValue = async () => {} // per widget
this.serialize_widgets = false // whole node

// after
node.widgets.add({ type: 'textarea', name: 'text', serialize: false })
node.serializesWidgets = false
```

Both are wire-format switches, so a conversion that drops them changes what the
saved workflow contains. `serialize` is orthogonal to `hidden` and `disabled`:
a widget can be visible and unsaved, or hidden and saved.

`add` throws if the name is already taken — rebuild is remove-then-add, and the
throw catches the common bug of appending a duplicate every execution.

**Do not convert a remove-and-recreate into `remove` plus a bare
`ComfyWidgets.*` call.** That leaves the pack on the unpublished surface, which
is the entire thing the conversion exists to end. If the widget it needs has no
`add` equivalent, that is an `api-gap` punt, not a partial conversion.

## Never let a write vanish

A handle lookup can fail. Reached through `?.`, a _write_ behind it does
nothing at all and the pack cannot tell:

```js
// ✗ silently does nothing when the node has not joined a graph yet — which is
//   exactly when a helper like this is usually called
comfy.graph.node(String(node.id))?.widgets.get(name)?.setHidden(true)

// ✓ use the handle you were given
function hideForGood(node, name) {
  node.widgets.get(name)?.setHidden(true) // reads may be optional
}
```

`comfy.graph.node(id)` resolves through the graph, so it returns nothing for a
node that has not been added yet. If you find yourself looking a node up by id
inside a helper, the helper should be taking a `NodeHandle` instead — its
caller has one.

Optional chaining on a _read_ is fine: `?.getValue()` returning undefined is
visible. On a write it is a bug that never reports itself.

## Reordering

```js
node.widgets.reorder(['prompt', 'seed', 'steps']) // full permutation
node.widgets.move('prompt', 0) // single move
```

`reorder` **throws on a partial list** rather than dropping the widgets you
omitted — which is precisely how the splice idiom lost them. The error names
what is missing.

## `setOption` preserves accessors — do not hand-merge

```js
// kjnodes builds dynamic combos with a live getter
Object.defineProperty(
  newOpts,
  'values',
  Object.getOwnPropertyDescriptor(comboOptions, 'values')
)
```

`setOption` merges by **property descriptor**, so getters stay getters. If you
hand-roll `{ ...widget.options(), ...patch }` you will invoke the getter and
freeze its result — pinning a dynamic combo to a one-time snapshot, silently.

`options()` returns a frozen _snapshot_ by design; use `setOption` to write.

## `getCustomWidgets` — 91 packs

Returning plain objects the store never sees. The widget is mounted on the node
that needs it instead of registered as a global type:

```js
node.widgets.mount({
  name: 'slider',
  height: 40,
  render(container) {
    /* build DOM; call widget.setValue on change */
  },
  destroy() {
    /* release listeners, timers, observers */
  }
})
```

`render` receives the container and gets a plain DOM element, deliberately:
packs bundle their own Vue since ADR 0005, and a component from a foreign Vue
instance cannot be mounted. A render function is framework-agnostic and
sidesteps the dual-instance problem.

There is no global widget-type registry. If a pack genuinely needs one type
reused across many node types, mount it from a shared helper — that is the
supported shape, not a workaround.

## Traps summary

| Trap                                                                    | Consequence                                      |
| ----------------------------------------------------------------------- | ------------------------------------------------ |
| Treating same-index splice as a reorder                                 | Nonsense conversion of a cache-invalidation hack |
| `hidden = true` alone, where the old code also suppressed serialization | Wire-format change                               |
| Hand-merging options                                                    | Live getters flattened; dynamic combos freeze    |
| Assigning a new `widgets` array                                         | Renderer stops tracking                          |
| Assigning `widgets.length`                                              | Widget teardown skipped                          |
| Partial `reorder` list                                                  | Throws — by design; supply every name            |

## Source data

Counts from the registry census at `~/comfy/nodes-compat-study/`
(`results/registry_scan.json`, 4,969 packs). Grep-derived with a known
false-positive rate — sample-verify before citing an individual number.
