# Magic Patch CLI

Magic Patch converts a pristine or partially converted ComfyUI custom-node
pack into a new, complete V2 pack folder. It delegates implementation work to
an already-installed and authenticated Codex or Claude Code CLI, then applies
deterministic local acceptance gates. Comfy does not hold the contributor's
model credentials or pay for the conversion inference.

## Prerequisites

- Python 3.13 for the current Comfy runtime and V2 contract.
- Either Codex CLI or Claude Code, installed, authenticated, and usable from
  the current shell:
  - **Codex CLI:** follow OpenAI's current
    [Codex CLI setup](https://learn.chatgpt.com/docs/codex/cli). On macOS or
    Linux, its documented standalone install is
    `curl -fsSL https://chatgpt.com/codex/install.sh | sh`. Run `codex` in a
    project directory and choose **Sign in with ChatGPT** (or another offered
    sign-in method) the first time it starts. OpenAI documents all supported
    methods on the [Codex authentication page](https://learn.chatgpt.com/docs/auth).
  - **Claude Code:** follow Anthropic's current
    [Claude Code setup guide](https://code.claude.com/docs/en/getting-started).
    Its recommended macOS/Linux/WSL native install is
    `curl -fsSL https://claude.ai/install.sh | bash`. Run `claude` and follow
    the browser login prompts. `claude --version` and `claude doctor` verify
    the installation. Anthropic documents account and provider choices on the
    [Claude Code authentication page](https://code.claude.com/docs/en/authentication).
- Node.js on `PATH` when converting frontend JavaScript, for an additional
  syntax check.
- `git` and an authenticated `gh` CLI only when using `--create-pr`.
- A ComfyUI checkout containing `comfy_api` and `nodes.py` to prove the result
  loads through the normal local V2 entrypoint. The current checkout is used
  automatically; override it with `--core-root` or `COMFY_CORE_ROOT`.
- Optionally, an executable implementing the Magic Patch verifier protocol.
  The Secure Nodes project can provide `comfy-secure-verify-pack`, but Magic
  Patch does not import or require that project.

The command invokes only the contributor's ambient CLI login. It does not read
an API key or call a model API directly. Run `codex` or `claude` once and
finish its login before starting Magic Patch.

## Convert a pack

From this repository:

```bash
export COMFY_CORE_ROOT=/path/to/ComfyUI

/path/to/python3.13 -m tools.magic_patch \
  /path/to/Original-Pack \
  /path/to/Original-Pack-converted \
  --agent auto
```

`auto` prefers Codex and falls back to Claude Code. Select one explicitly with
`--agent codex` or `--agent claude`; use `--model` only when a provider-specific
override is needed.

The input is never modified. The output path must not already exist. Magic
Patch creates a sibling staging directory, clones the whole original pack into
`v2/`, overlays any existing V2 draft, and permits the agent to edit only that
staged copy.

Patch identity is the pack's upstream Git commit: `x` plus the first seven
commit characters. When the input folder is the root of its Git checkout,
Magic Patch discovers it automatically. For an extracted archive or a pack
nested inside a monorepo, pass the pinned upstream commit explicitly with
`--source-sha`; use `--pack-slug` when the registry slug cannot be derived from
the input folder name.

A successful run publishes four sibling artifacts as one no-overwrite set:

```text
Original-Pack-converted/
  ... unchanged original files ...
  v2/
    ... complete converted pack ...
    comfy-api.pyi
    comfy-api.d.ts
    pyproject.toml
    secure-nodes.json
    V2_CONVERSION.md
Original-Pack-converted.zip
Original-Pack-converted.patches/
  original-pack-x1a2b3c4.json
  original-pack-x1a2b3c4.diff
Original-Pack-converted.magic-patch.json
```

The ZIP contains exactly one top-level `Original-Pack-converted/` folder with
the pristine source files at its root and the complete conversion under its
`v2/` child. It can be submitted directly as a V2 pack. ZIP member order,
timestamps, contents, and modes are deterministic. Override its destination
with `--pack-zip`, or omit it with `--no-pack-zip`.

The `.patches/` directory contains the same `.json` manifest and reviewable
`.diff` pair used by the backend deployment path. Magic Patch applies the pair
to a fresh copy of the original and requires the result to match the published
`v2/` tree byte-for-byte before it exposes any artifact. Override that directory
with `--patch-output`.

On failure, no output folder, ZIP, report, or patch directory is published.
The command prints the preserved staging path, containing agent logs and
`FAILURE.txt`, so a subsequent run can be diagnosed without losing evidence.

Use `--dry-run` to check paths, bundled contracts, the selected agent, and PR
prerequisites without invoking a model.

## Agent loop and acceptance gates

Each pass receives the complete pack plus trusted Python/frontend conversion
guidance. It returns a schema-constrained census and test record. Deterministic
findings become the repair prompt for the next pass, up to `--max-passes`.

Publishing requires all of the following:

- the original pack tree is byte-for-byte and mode-for-mode unchanged;
- pack-owned `AGENTS.md`, `CLAUDE.md`, `.agents/`, `.claude/`, and `.codex/`
  content was not exposed as agent control input and is restored correctly;
- `v2/` is a complete pack with no symlinks, caches, or nested `v2/`;
- the bundled published `.pyi` and `.d.ts` contracts are unchanged;
- Python parses and has no ambient ComfyUI imports;
- declared frontend JavaScript parses when Node.js is available and contains
  no known legacy host surfaces;
- `secure-nodes.json` is safe and agrees with `pyproject.toml`;
- the normal local ComfyUI V2 loader registers exactly the manifest node ids;
- the optional secure verifier passes when it is installed;
- a generated JSON/diff pair recreates the complete `v2/` tree byte-for-byte;
- the agent census has zero pending items and lists passing tests.

Source code remains untrusted data, but a coding agent necessarily reads it.
The normal local V2 load executes the converted entrypoint in a child process;
it is not an operating-system security boundary. Run deliberately hostile
repositories in a disposable machine or use the optional secure verifier.

## Optional secure-sandbox verification

Sandbox verification is `auto` by default. Magic Patch looks for
`comfy-secure-verify-pack` on `PATH`, or for the command named by
`COMFY_MAGIC_PATCH_SANDBOX_VERIFIER`. If no verifier is installed, all public
validation still runs and conversion can succeed. The public utility never
imports the secure runtime.

Use `--sandbox-verification required` when publication must have sandbox
evidence, or `--sandbox-verification off` to skip discovery. An explicit
verifier can be selected with `--sandbox-verifier /path/to/command`; its time
limit is controlled by `--sandbox-timeout`.

The public utility and optional verifier communicate through versioned JSON
request and result files. A discovered verifier that crashes, returns malformed
evidence, or reports an escape blocks publication even in `auto` mode. Passing
dynamic checks is evidence that the exercised imports and operations remained
inside the sandbox; it is not a mathematical proof that every possible code
path is incapable of escaping.

Magic Patch invokes the verifier without a shell:

```text
comfy-secure-verify-pack --request REQUEST.json --output RESULT.json
```

The request format is `comfy-magic-patch-verifier-request/1` and supplies
absolute `pack`, `source`, optional `core_root`, and `python_executable` paths.
The verifier must write `comfy-magic-patch-verifier-result/1` with a non-empty
`verifier` name, `status` equal to `passed`, `failed`, or `unavailable`, and
string arrays named `checks` and `errors`. Failed and unavailable results
require at least one error; passing results may not contain errors. In `auto`
mode, an unavailable platform backend is treated like an uninstalled verifier;
in `required` mode it blocks publication. This narrow protocol lets other
sandbox implementations integrate without coupling ComfyUI to private modules.

## Open a pull request

Add `--create-pr` to publish the validated `v2/` tree with the ambient `gh`
authentication:

```bash
/path/to/python3.13 -m tools.magic_patch \
  /path/to/upstream-pack-checkout \
  /path/to/upstream-pack-converted \
  --agent codex \
  --create-pr
```

Magic Patch discovers the GitHub repository and default branch from the source
checkout. It clones that repository into a disposable directory, checks out
the source revision when available, commits only the new or updated `v2/`
tree, pushes a generated branch, and opens a formatted PR against the original
repository. The body records the backend/frontend census, tests, validation
evidence, and conversion notes. The input checkout remains untouched.

If the authenticated user cannot push to the original repository, Magic Patch
creates or reuses their GitHub fork and opens a cross-fork PR. Direct the PR at
a different repository or a pack nested within a monorepo with:

```bash
--pr-repo owner/repository \
--pr-pack-path path/to/pack \
--pr-base main \
--pr-draft
```

`--pr-branch` and `--pr-title` override the generated branch and concise commit
title. PR publication happens only after local conversion succeeds. A GitHub
failure does not remove the converted pack or its local report.
