# Converting Python custom nodes to ComfyUI V2

This guide is portable conversion input. The only authoritative API surface is
the exact `v2/comfy-api.pyi` file bundled into the pack. Search that file before
using an `io` type, SDK ref, context domain, method, keyword, or enum. If it is
not declared there, it is not available.

## Non-negotiable structure

- Edit only the complete `v2/` tree prepared by Magic Patch.
- Keep one implementation tree with the upstream file and module layout.
- Do not create `secure_nodes/`, `v2_*.py` sidecars, secure aliases, or a second
  registration path.
- Keep every original node id, class name, display name, category, input id,
  output order, list flag, lazy behavior, validation rule, fingerprint rule,
  workflow wire value, and UI result unless the published API forces a change.
- Never append `Secure`, `V2`, a lock glyph, or other conversion branding to an
  identity used by workflows.
- Never import ambient host packages: `comfy`, `comfy_execution`,
  `folder_paths`, top-level `nodes`, or `server`.
- Do not import or execute code from the pristine pack root. Relative imports
  must resolve entirely inside `v2/`.

## Inventory before editing

Find every registration and record:

- `NODE_CLASS_MAPPINGS`, dynamic mapping updates, aliases, and display mappings;
- `INPUT_TYPES`, `RETURN_TYPES`, `RETURN_NAMES`, `OUTPUT_NODE`, `OUTPUT_IS_LIST`,
  `INPUT_IS_LIST`, `CATEGORY`, `DESCRIPTION`, `FUNCTION`, `VALIDATE_INPUTS`,
  `IS_CHANGED`, and `check_lazy_status`;
- imports that reach core, filesystem paths, network, subprocesses, model
  folders, caches, application globals, or server routes;
- optional dependencies and hardware-only branches;
- relative assets and any initialization side effects.

The final backend census counts original workflow node ids. Every id must be
supported, explicitly rejected for a policy reason, or pending because an API
is missing. Missing APIs are never policy rejections.

## V2 node form

V2 classes inherit `io.ComfyNode`, define a schema, and return
`io.NodeOutput`:

```python
from comfy_api.latest import io


class Example(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OriginalWorkflowId",
            display_name="Original Display Name",
            category="Original/Category",
            description="Original description",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("amount", default=1, min=0, max=64, step=1),
            ],
            outputs=[io.Image.Output("image")],
        )

    @classmethod
    async def execute(cls, image, amount) -> io.NodeOutput:
        result = image
        return io.NodeOutput(result)
```

Use the exact constructors and keywords from `comfy-api.pyi`. Common legacy
mappings are:

| Legacy | Published V2 form |
|---|---|
| `("IMAGE",)` | `io.Image.Input("id")` |
| `("MASK",)` | `io.Mask.Input("id")` |
| `("INT", {...})` | `io.Int.Input("id", ...)` |
| `("FLOAT", {...})` | `io.Float.Input("id", ...)` |
| `("STRING", {...})` | `io.String.Input("id", ...)` |
| `("BOOLEAN", {...})` | `io.Boolean.Input("id", ...)` |
| a fixed string option list | `io.Combo.Input("id", options=[...])` |
| an unpublished custom wire type | `io.Custom("EXACT_TYPE").Input("id")` |

Preserve optional, lazy, force-input, list, tooltip, display, default, min,
max, step, and multiline semantics. Hidden prompt metadata uses published
hidden schema declarations, never `server` or a global prompt object.

Translate legacy optional methods without weakening them:

- `VALIDATE_INPUTS` becomes `validate_inputs`;
- `IS_CHANGED` becomes `fingerprint_inputs`;
- lazy input selection remains `check_lazy_status`;
- a method must remain class-level or instance-stateful according to its real
  behavior. Do not turn retained cross-call state into a shared module global.

## Choose the narrowest execution mode

There are two V2 input modes. Absence of `SDK_REFS` does not make a node V1 or
prevent a secure runtime from sandboxing it.

### Ordinary guest values

Use ordinary V2 inputs when the node operates only on JSON-like values and
guest-owned tensors and its dependencies import in the API-only environment.
The runtime materializes ordinary tensor inputs inside the guest and wraps
ordinary tensor outputs back into host refs. Preserve the original algorithm;
do not add broker calls merely to make a conversion look secure.

This mode is appropriate for pure math, string/list transforms, tensor kernels,
and pack-local Python dependencies that need no host authority.

### Opaque SDK refs

Set `SDK_REFS = True` when an input is a live host object or the operation must
use a brokered host capability. Inputs then arrive as the exact ref types in
`comfy-api.pyi`, such as `sdk.ImageRef`, `sdk.MaskRef`, `sdk.LatentRef`,
`sdk.ModelRef`, `sdk.VaeRef`, or `sdk.ClipRef`.

Prefer bounded ref operations that keep data host-side:

```python
from comfy_api.latest import io, sdk


class Invert(io.ComfyNode):
    SDK_REFS = True

    @classmethod
    async def execute(cls, image: sdk.ImageRef) -> io.NodeOutput:
        return io.NodeOutput(await image.invert())
```

Use `await ref.raw()` or `await ref.value()` only when faithful guest compute
requires materialization and the class requests the exact published permission.
Use published constructors/wrappers for outputs; do not construct ref wire
tokens or reach into private attributes. Treat refs as execution-scoped: do not
cache them in module globals, files, closures, or class state.

Host-wide work goes through the closed domains on `sdk.ctx()`—models, assets,
interaction, integration adapters, execution, and similar surfaces declared in
the contract. Inputs are bounded names and scalar options, never arbitrary
filesystem paths, module names, callbacks, object paths, or source text. A
permission declaration asks for authority; it does not grant it.

When a required operation is missing, keep the node id registered only if it
can fail clearly without weakening validation or importing host internals.
Record the missing operation as an API gap and return pending.

## Pure and third-party dependencies

Pack-local pure helpers may remain unchanged when they import cleanly without
ComfyUI core. Convert imports to relative package imports where needed.

For code previously borrowed from `comfy.utils` or another core module:

1. Look for an equivalent published SDK/ref operation.
2. If the function is genuinely small, pure, stable, and license-compatible,
   move only that algorithm into a clearly pack-local helper and test it
   differentially.
3. Otherwise request an API gap. Never copy large host subsystems into a pack
   and never expose a general “call core function” broker.

Dependencies belong in `v2/pyproject.toml`. Preserve upstream package/version
requirements and declare the selected Python minor exactly, for example:

```toml
[project]
name = "upstream-pack-name"
version = "0.0.0"
requires-python = ">=3.13,<3.14"
dependencies = [
  "example-package>=1,<2",
]
```

Do not create a venv, install requirements, or download models during
conversion. Installation is a later per-pack deployment phase.

## Assets, paths, and side effects

The `v2/` tree is a full pack clone, so normal package-relative and
`__file__`-relative reads resolve inside it. Keep fonts, lookup tables, model
metadata, and other unchanged assets at the same relative paths. Do not embed
absolute developer paths.

Replace filesystem and network authority with published SDK declarations and
operations. Pack initialization must not register host routes, mutate global
ComfyUI state, inspect the host filesystem, or launch processes. Static/web
directories are data declarations consumed by the trusted loader.

## Registration and manifest

Replace legacy `NODE_CLASS_MAPPINGS` and display mappings with a normal V2
`comfy_entrypoint` returning a `ComfyExtension`. Its node list contains the
converted classes, whose schemas preserve the original workflow ids. Do not
leave the legacy registration names in `v2/__init__.py`: the local loader gives
them precedence over `comfy_entrypoint`.

Also create `v2/secure-nodes.json`. Normal local ComfyUI uses the V2 entrypoint;
an optional secure runtime reads this metadata without importing the pack in
the host. Its top-level form is:

```json
{
  "format": "comfy-secure-nodes-v1",
  "nodes": {
    "OriginalWorkflowId": {
      "class": "Example",
      "module": "nodes.example",
      "methods": {
        "check_lazy_status": false,
        "fingerprint_inputs": false,
        "validate_inputs": false
      },
      "permissions": [],
      "schema": { "attrs": {}, "hidden": [], "inputs": [], "outputs": [] },
      "sdk_refs": false
    }
  },
  "runtime": {
    "python": { "requires": ">=3.13,<3.14", "resolved": "3.13" }
  },
  "web_directory": "web"
}
```

Encode every schema field, input, output, hidden input, enum value, list flag,
and method declaration faithfully. The node key and `schema.attrs.node_id`
must be the original id. `module` is relative to `v2/` and must point to the
real converted source. `class` is its real class name. Set `web_directory` to
`null` when absent. Include frontend permissions, required weights, static
directories, asset directories, and scheduler providers only when the pack
actually declares them through the published format.

## Behavior-driven verification

For each meaningful behavior:

1. Identify the observable contract: output tensors/values, UI payload,
   validation, fingerprint, lazy selection, expansion, errors, and state.
2. Run the original behavior in a controlled local test when safe.
3. Run the converted implementation on identical inputs.
4. Compare values and decoded tensors, not incidental encodings.
5. Prove the converted module imports with only the pack and published API
   available. `import comfy`, `folder_paths`, `nodes`, and `server` must fail.
6. Exercise optional/hardware branches with real dependencies when available
   and truthful fakes otherwise. Label hardware-only evidence precisely.

Use varied inputs so caching cannot impersonate execution and assert that
transforming nodes actually differ from their input. Test every branch that
selects a kernel, dtype, layout, fallback, model family, or optional
dependency. Never turn a missing implementation into a silent identity result.

Finally verify the complete pack census, `secure-nodes.json`, normal V2
registration, and all relative assets. When a secure verifier is available,
also verify guest import and execution. Record commands and outcomes in
`v2/V2_CONVERSION.md`.

## Do not

- edit the pristine root;
- create a sidecar or alternate secure registration tree;
- rename workflow ids or change schema wire types;
- import any ambient ComfyUI module, even lazily;
- smuggle buffers, paths, objects, callbacks, or arbitrary operations across
  the RPC boundary;
- weaken validation to make a test pass;
- stub a required algorithm with identity output;
- claim unsupported hardware or optional-dependency behavior was tested;
- download dependencies, weights, or source during conversion;
- claim completion while any discovered node or API gap remains pending.
