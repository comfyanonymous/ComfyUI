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
  compatibility branches that are no longer needed.
- Revert or disable problematic behavior quickly when it breaks users. It is
  better to remove a broken feature path than keep a complicated partial fix.
- Preserve existing APIs, node names, model-loading behavior, file layout, and
  workflow compatibility unless the change is explicitly about replacing them.
- Code must look hand-written for this repository. Changes that read like
  generic AI-generated code will be rejected automatically: unnecessary helper
  layers, vague names, boilerplate comments, defensive branches without a real
  failure mode, broad rewrites, or code that ignores the local style.

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
- Keep comments sparse and useful. Short TODOs are fine when they name the
  concrete missing follow-up.

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
- Prefer one coherent behavioral change per commit. Dependency pins, tests, and
  the code that needs them may be in the same commit when they are inseparable.
- In reviews, prioritize real user impact: crashes, wrong dtype/device behavior,
  memory regressions, broken model loading, workflow incompatibility, and noisy
  or misleading user-facing output.
