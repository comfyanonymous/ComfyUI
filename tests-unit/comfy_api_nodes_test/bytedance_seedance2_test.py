import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy_api_nodes.apis.bytedance import TaskTextContent  # noqa: E402
from comfy_api_nodes.nodes_bytedance import (  # noqa: E402
    _seedance2_build_request,
    _seedance2_price_badge,
)

SEEDANCE_25 = "dreamina-seedance-2-5-260628"


def build(**overrides):
    """Build a Seedance 2.x request from the widget values a node would collect."""
    model = {
        "prompt": "continue the shot",
        "resolution": "720p",
        "ratio": "16:9",
        "duration": 8,
        "generate_audio": True,
        "output_format": "mp4",
    }
    model.update(overrides)
    return _seedance2_build_request(
        model, SEEDANCE_25, [TaskTextContent(text="continue the shot")], 0, False, ratio=model["ratio"]
    )


@pytest.mark.parametrize(
    "task_type,expected_ratio,expected_duration,expected_field",
    [
        # An absent key is the text-to-video and first/last-frame path: they share this
        # builder but have no task_type widget, so nothing may be coerced or sent.
        (None, "16:9", 8, None),
        ("auto", "16:9", 8, None),
        ("reference", "16:9", 8, "reference"),
        # An edit inherits both the aspect ratio and the length of the source clip.
        ("edit", "adaptive", -1, "edit"),
        # An extension inherits only the aspect ratio; the requested length survives.
        ("extend", "adaptive", 8, "extend"),
    ],
)
def test_task_type_coercion(task_type, expected_ratio, expected_duration, expected_field):
    request = build() if task_type is None else build(task_type=task_type)
    assert request.ratio == expected_ratio
    assert request.duration == expected_duration
    assert request.omni_reference_task_type == expected_field


def test_extend_preserves_requested_duration():
    """Duration is the only constraint separating extend from edit, so it must survive."""
    assert build(task_type="extend", duration=27).duration == 27


def test_seedance_20_omits_the_field():
    """omni_reference_task_type is a 2.5 field; the 2.0 inputs carry no task_type widget."""
    assert build().omni_reference_task_type is None


def test_price_badge_tracks_task_type():
    """Both the dependency list and the JSONata must move together or the badge goes stale."""
    badge = _seedance2_price_badge(with_reference_videos=True)
    assert "model.task_type" in badge.depends_on.widgets
    assert 'model.task_type") = "edit"' in badge.expr
