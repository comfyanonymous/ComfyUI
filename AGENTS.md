# AGENTS.md

## Scope

These instructions apply to the entire repository unless a deeper AGENTS.md overrides them.

## Project context

This is a standard Git-based ComfyUI repository on Windows.

Repository root:
`C:\AI_PROJECTS\ComfyUI`

Primary Python environment:
`C:\AI_PROJECTS\ComfyUI\venv\Scripts\python.exe`

Primary shell:
Windows PowerShell

## Core working rules

- Preserve existing functionality unless the user explicitly asks for redesign or removal.
- Do not remove features, nodes, scripts, launchers, or config behavior unless explicitly told.
- Fix root causes instead of patching symptoms where practical.
- Reuse and improve existing logic instead of duplicating it.
- Do not invent files, APIs, packages, results, or test outcomes.
- Do not claim something was tested unless you actually ran the test or command.
- Keep changes tightly scoped to the user’s request.
- Stay focused on the current task. Do not perform unrelated cleanup or refactors.
- Prefer additive, reversible changes.

## Environment rules

- Always use the repo-local Python interpreter:
  `C:\AI_PROJECTS\ComfyUI\venv\Scripts\python.exe`
- Never use a global Python, Conda environment, Windows Store Python, or some other interpreter unless the user explicitly asks.
- When installing packages, use:
  `C:\AI_PROJECTS\ComfyUI\venv\Scripts\python.exe -m pip ...`
- Do not use bare `pip` commands.
- Assume the repo-local venv is the source of truth unless the user says otherwise.

## Command rules

When giving commands or running commands:

- Use Windows PowerShell syntax.
- Always state the working directory first.
- Prefer exact commands over summaries.
- Prefer commands that are safe to copy/paste.

## File safety rules

- Do not modify files outside this repository unless the user explicitly asks.
- Do not edit anything inside `venv\Lib\site-packages` unless the user explicitly asks for a local hotfix there.
- Do not delete models, outputs, checkpoints, LoRAs, embeddings, or user assets unless explicitly asked.
- Be careful with files under:
  - `models\`
  - `custom_nodes\`
  - `user\`
  - launch scripts
  - config files
- Before changing behavior in `custom_nodes`, inspect that node’s current imports, dependencies, and startup assumptions.

## Git rules

If modifying files:

- Work on the current branch unless the user explicitly asks for a new one.
- Do not rewrite history.
- Do not amend existing commits unless the user explicitly asks.
- Do not create commits unless the user asks for a commit.
- Show `git status` after making changes when relevant.

## Validation rules

After code changes, run the smallest relevant validation that actually checks the change.

Preferred validation commands:

- Python syntax check for a file:
  `C:\AI_PROJECTS\ComfyUI\venv\Scripts\python.exe -m py_compile <file>`
- Package import check:
  `C:\AI_PROJECTS\ComfyUI\venv\Scripts\python.exe -c "import <module>; print('ok')"`
- If the change affects startup or node imports, use the repo-local environment and validate with the relevant launch command or targeted import path when practical.
- If a full startup test is too heavy, say so clearly and run the best targeted validation available.

## Reporting rules

In responses:

- State exactly which files were changed.
- State exactly which commands were run.
- State exactly what was verified and what was not verified.
- If something is risky, incomplete, or untested, say that plainly.
- Do not hide uncertainty.

## ComfyUI-specific guidance

- Treat startup logs as important evidence.
- When debugging node import failures, identify whether the problem is:
  1. missing package
  2. wrong package version
  3. broken path/import
  4. incompatible ComfyUI/custom-node API change
  5. GPU/CUDA/Torch mismatch
- For package fixes, prefer minimal changes over large dependency churn.
- If a custom node shells out to `pip`, prefer replacing that with interpreter-safe commands using `python -m pip` when asked to fix it.
- Be cautious about torch, CUDA, TensorRT, xformers, flash-attn, triton, sageattention, and custom kernel packages because version mismatches can destabilize the whole stack.

## Instruction priority

- Follow direct user instructions first.
- Then follow deeper nested AGENTS.md files if present.
- Then follow this root AGENTS.md.
