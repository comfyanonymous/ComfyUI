import torch
from unittest.mock import MagicMock, patch

from comfy.cli_args import args

args.cpu = True

import latent_preview  # noqa: E402


def _run_steps(interval, steps):
    previewer = MagicMock()
    previewer.decode_latent_to_preview_image.return_value = ("JPEG", None, 512)
    model = MagicMock()
    with patch.object(latent_preview, "get_previewer", return_value=previewer), \
            patch.object(latent_preview.comfy.utils, "ProgressBar"), \
            patch.object(args, "preview_interval", interval):
        callback = latent_preview.prepare_callback(model, steps)
        x0 = torch.zeros(1, 4, 8, 8)
        for step in range(steps):
            callback(step, x0, x0, steps)
    return previewer.decode_latent_to_preview_image.call_count


def test_default_previews_every_step():
    assert _run_steps(1, 10) == 10


def test_interval_skips_steps_and_keeps_last():
    # steps 0, 3, 6, 9 -> 4 decodes, step 9 is also the last step
    assert _run_steps(3, 10) == 4


def test_last_step_always_previewed():
    # steps 0, 4, 8 plus the last step 9
    assert _run_steps(4, 10) == 4
