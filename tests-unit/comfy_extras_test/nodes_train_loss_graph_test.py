"""LossGraphNode regression tests for a loss series with no spread.

`LossGraphNode.execute` normalizes the curve with

    (l - min_loss) / (max_loss - min_loss)

which raises `ZeroDivisionError` whenever every recorded loss is identical.
A `steps=1` run hits that by definition -- it has exactly one value -- and a
one-step run is the cheap way to validate a training config before committing
to a long one. Training itself completes and the LoRA is written; only the
graph node fails, so the run is lost at its last step.

Assertions read the PNG the node actually rendered rather than recomputing the
scaling here, so a test cannot pass against a reimplementation of the bug.
"""

import glob
import os

import pytest
import torch
from PIL import Image

from comfy.cli_args import args as cli_args

# comfy.model_management resolves the torch device at import time and asks CUDA
# for it unless args.cpu is set, so this has to happen before nodes_train is
# imported. Same guard the SeedVR2 node tests use.
if not torch.cuda.is_available():
    cli_args.cpu = True

import folder_paths  # noqa: E402
from comfy_extras.nodes_train import LossGraphNode  # noqa: E402

HEIGHT = 480  # plot area height inside the rendered canvas


@pytest.fixture
def render(tmp_path):
    """Render a loss series and hand back the rows its blue line occupies.

    PreviewImage writes the figure to the temp directory, so point that at
    tmp_path and read the result back from there.
    """
    original = folder_paths.get_temp_directory()
    folder_paths.set_temp_directory(str(tmp_path))

    def _render(loss_values):
        result = LossGraphNode.execute(
            loss={"loss": loss_values}, filename_prefix="loss_graph"
        )
        assert result is not None
        written = glob.glob(os.path.join(str(tmp_path), "**", "*.png"), recursive=True)
        assert written, "node did not write a preview image"
        image = Image.open(written[0]).convert("RGB")
        width, _ = image.size
        rows = {
            index // width
            for index, (r, g, b) in enumerate(image.getdata())
            if b > 200 and r < 100 and g < 100
        }
        return sorted(rows)

    yield _render
    folder_paths.set_temp_directory(original)


def test_single_step_run_renders_instead_of_raising(render):
    """steps=1 gives exactly one loss value, so min == max by definition."""
    render([3.1607])


def test_constant_loss_series_renders_instead_of_raising(render):
    """Not only steps=1 -- any run whose losses are all equal hit the divide."""
    render([2.0, 2.0, 2.0])


def test_zero_loss_series_renders_instead_of_raising(render):
    """min == max == 0, so numerator and divisor are both zero."""
    render([0.0, 0.0])


@pytest.mark.parametrize("value", [2.0, 5.0, 0.25])
def test_constant_series_is_drawn_flat(render, value):
    """A constant series plots as a flat line at the same height whatever the
    constant is -- the only true statement about it is that it did not move.

    Mid-height rather than 0.0 or 1.0 deliberately: those pin the line to the
    bottom or top axis, which reads as "lowest" or "highest" loss.
    """
    rows = render([value] * 4)
    assert rows, "no line was drawn"
    assert max(rows) - min(rows) <= 2, "a constant series must not slope"
    assert abs(sum(rows) / len(rows) - HEIGHT / 2) <= 2, "should sit at mid-height"


def test_varying_series_still_spans_the_plot(render):
    """Regression guard: the normal path keeps using the full plot height."""
    rows = render([3.0, 2.0, 1.0])
    assert min(rows) <= 2, "highest loss should reach the top of the plot"
    assert max(rows) >= HEIGHT - 2, "lowest loss should reach the bottom"
