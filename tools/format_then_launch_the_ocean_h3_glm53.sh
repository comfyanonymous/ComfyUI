#!/usr/bin/env bash
set -euo pipefail
PYTHON=/home/op/ai/minimax-h3/.venv-story/bin/python
ROOT=/home/op/ai/ComfyUI
"$PYTHON" "$ROOT/tools/prepare_the_ocean_glm53_sentence_prompts.py"
systemd-run --user \
  --unit=the-ocean-h3-glm53-render-20260816.service \
  --collect \
  --property=MemoryMax=infinity \
  --property=TimeoutStartSec=infinity \
  "$PYTHON" "$ROOT/tools/run_the_ocean_h3_glm53_sentence_series.py"
