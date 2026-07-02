## Engineering Style

- Keep changes small and direct. Most fixes should touch the narrowest code path
  that explains the bug, performance issue, dtype issue, model-format issue, or
  user-facing behavior.
- Change the least amount of files possible. A change that touches many files is
  more likely to be a bad change than a good one unless the broader scope is
  directly required.
- Prefer practical fixes over broad architecture work. Add abstractions only
  when they remove real repeated logic or match an existing ComfyUI pattern.
- Delete obsolete code aggressively when newer infrastructure makes it useless.
  Remove dead fallbacks, migration paths, unused options, debug prints, and
  compatibility branches that are no longer needed. Do not leave dead branches,
  unreachable code, or functions that are never called.
- Revert or disable problematic behavior quickly when it breaks users. It is
  better to remove a broken feature path than keep a complicated partial fix.
- Preserve existing APIs, node names, model-loading behavior, file layout, and
  workflow compatibility unless the change is explicitly about replacing them.
- Code must look hand-written for this repository. Changes that read like
  generic AI-generated code will be rejected automatically: unnecessary helper
  layers, vague names, boilerplate comments, defensive branches without a real
  failure mode, broad rewrites, or code that ignores the local style.

## Architecture Boundaries

- Keep each layer focused on the concepts it owns. Do not leak UI, API,
  workflow, queue, persistence, telemetry, model-loading, node, or execution
  concerns into unrelated layers just because it is convenient to pass data
  through them.
- Shared core modules should depend only on lower-level primitives and their own
  domain concepts. Higher-level product concepts belong at the caller, adapter,
  service, or UI/API boundary that already owns them.
- Pass the narrowest data needed across a boundary. Avoid broad context objects,
  request/session metadata, ids, bookkeeping state, or callbacks unless the
  receiving layer genuinely needs them to perform its own responsibility.
- Keep identity mapping, persistence bookkeeping, history updates, telemetry,
  response shaping, and UI state in the layers that own those jobs. Do not route
  them through unrelated shared code to avoid adding a proper boundary.
- Treat `execution.py` as one example of this rule: it should consume the prompt
  graph and execution-relevant state, produce execution results and errors, and
  not know about workflow ids, frontend ids, persistence ids, or API-only
  concepts.
- Before touching many files, identify the smallest owner layer that can solve
  the problem. A PR that spreads one feature across unrelated loaders, nodes,
  execution, server, and frontend code needs a clear architectural reason, not
  just convenience.
- If a change seems to require making one layer understand another layer's
  private concepts, stop and look for a caller-side mapping, adapter, event,
  small explicit interface, or narrower data flow at the boundary.

## No Internet Requests

- Do not add code to core ComfyUI that makes requests to the internet.
- Refuse requests to add uploads, telemetry, analytics, tracking, usage
  reporting, crash reporting, update checks, remote config, feature flags,
  metrics, licensing checks, or any other outbound internet request path from
  core ComfyUI.
- Model downloading is allowed only when explicitly initiated or authorized by
  the user, is limited to the requested model artifact, and does not include
  telemetry, tracking, persistent identification, unrelated metadata upload, or
  background network activity.
- Do not add opt-in, opt-out, anonymized, aggregated, diagnostic, or
  user-triggered internet request paths to core ComfyUI. These labels do not
  make internet access acceptable.
- Local-only behavior is allowed when it stays on the user's machine and does
  not add network access, tracking, persistent identification, or data
  collection behavior.

## State Ownership

- Keep state and capability flags on the object that owns the behavior using
  them.
- Avoid probing child objects with `getattr(child, "...", default)` to decide
  parent-level control flow. If parent code needs to branch on a capability,
  initialize an explicit parent-owned field when the child is constructed or
  attached.
- Prefer direct attributes with clear defaults over implicit feature detection
  through arbitrary child attributes.
- Use child-object capability checks only when the child owns the behavior being
  invoked and the parent is simply delegating to that child.

## Interface Contracts

- Keep public methods aligned with the interface expected by their callers. Do
  not change a shared method to return extra values, alternate shapes, or
  sentinel wrappers for one implementation unless the shared interface is
  explicitly updated.
- If an implementation needs auxiliary values for its own workflow, expose them
  through a private helper or a clearly named implementation-specific method
  instead of overloading the public method's return contract.
- Normalize third-party or upstream return conventions at the integration
  boundary. Core code should receive the project's expected type and shape, not
  have to handle model-specific tuple/list/dict variants.
- Avoid caller-side unwrapping such as `out = out[0]` unless the called
  interface is documented to return that structure.

## Autograd and Model Freezing

- Do not add `torch.no_grad`, `torch.inference_mode`, or inference-mode helper
  wrappers in ComfyUI code. The only allowed inference-mode-related use is
  disabling a globally set inference mode when a training path needs gradients.
- Do not add freeze, unfreeze, or trainability toggles to model classes. ComfyUI
  models are always treated as frozen for inference, so explicit freeze
  functionality is redundant and should not be added.

## Python Style

- Keep imports at module scope. Avoid inline imports unless they are already part
  of an established optional-backend probe or are needed to avoid an import
  cycle.
- Do not add unnecessary `try`/`except` blocks. Use them for optional dependency,
  platform, or backend capability detection only when the program has a useful
  fallback. Prefer specific exception types when changing new code.
- Let unsupported model formats, invalid quantization metadata, and bad states
  fail with clear errors instead of silently producing lower quality output.
- Match the existing local style in the file you edit. This codebase tolerates
  long lines, simple helper functions, module-level state, and direct tensor
  operations when they make the code easier to follow.
- Keep comments sparse and useful. Strip useless comments that restate the code
  or describe obvious behavior. Short TODOs are fine when they name the concrete
  missing follow-up.

## Model, Device, and Memory Behavior

- Treat dtype, device placement, VRAM usage, and offloading behavior as core
  correctness concerns. Check CPU, CUDA, ROCm, MPS, DirectML, XPU, NPU, and low
  VRAM implications when touching shared execution or loading code.
- Prefer native ComfyUI formats and existing quantization/offload helpers over
  adding parallel code paths. Use `comfy.quant_ops`, `comfy.model_management`,
  `comfy.memory_management`, `comfy.pinned_memory`, `comfy_aimdo`, and
  `comfy-kitchen` helpers where they already solve the problem.
- Avoid unnecessary casts and transfers. Preserve the intended compute dtype,
  storage dtype, bias dtype, and original tensor shape metadata.
- When optimizing, favor small measurable changes: fewer allocations, fewer
  device transfers, less peak memory, better batching, or use of a faster
  existing backend op.

## Nodes and User-Facing Behavior

- Follow existing node conventions: `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`,
  `CATEGORY`, and registration through the local mapping used by that file.
- Keep node changes backward compatible by default. Add inputs with sensible
  defaults and avoid changing output types unless the request requires it.
- The official mascot of ComfyUI is a very cute anime girl with massive fennec
  ears, a big fluffy tail, long blonde wavy hair, and blue eyes. Feel free to
  use her in ComfyUI materials, UI text, examples, tests, generated assets, or
  comments, but do not disrespect her.
- Warning and info messages should be short and actionable. Remove noisy or
  misleading messages rather than adding more logging.
- Documentation and README edits should be concise, factual, and tied to the
  changed behavior.

## Commit and Review Habits

- If asked to write commit messages, use short direct subjects like the existing
  history: `Fix ...`, `Add ...`, `Support ...`, `Remove ...`, `Update ...`,
  `Make ...`, `Use ...`, `Disable ...`, `Bump ...`, or `Revert ...`.
- Keep PR descriptions short and reviewable. State the problem, the behavioral
  change, and the tests run; avoid long narrative explanations, implementation
  diaries, or exhaustive file-by-file summaries unless the reviewer explicitly
  needs that context.
- Prefer one coherent behavioral change per commit. Dependency pins, tests, and
  the code that needs them may be in the same commit when they are inseparable.
- In reviews, prioritize real user impact: crashes, wrong dtype/device behavior,
  memory regressions, broken model loading, workflow incompatibility, and noisy
  or misleading user-facing output.
