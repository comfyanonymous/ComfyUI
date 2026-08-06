import pytest
import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy.ldm.wan import vae as wan21  # noqa: E402
from comfy.ldm.wan import vae2_2 as wan22  # noqa: E402


class _ChunkModule(nn.Module):
    def __init__(self, chunks):
        super().__init__()
        self.chunks = chunks
        self.index = 0

    def forward(self, _x, **_kwargs):
        chunk = self.chunks[self.index]
        self.index += 1
        return chunk


def _make_vae(vae_class, **modules):
    vae = vae_class.__new__(vae_class)
    nn.Module.__init__(vae)
    for name, module in modules.items():
        setattr(vae, name, module)
    return vae


def _chunk(value, channels=2, frames=1):
    values = torch.arange(channels * frames, dtype=torch.float64)
    return values.reshape(1, channels, frames, 1, 1) + value


def _encoded_output(tensor):
    return tensor.chunk(2, dim=1)[0]


def _decoded_output(tensor):
    return wan22.unpatchify(tensor, patch_size=2)


_ASSEMBLY_CASES = [
    pytest.param(
        wan21,
        wan21.WanVAE,
        "encoder",
        "conv1",
        "encode",
        [
            (torch.zeros(1, 1, 9, 1, 1, dtype=torch.float64), [_chunk(10), None, _chunk(20), _chunk(30, frames=2), None]),
            (torch.zeros(1, 1, 1, 1, 1, dtype=torch.float64), [_chunk(40)]),
        ],
        _encoded_output,
        id="wan21-encode",
    ),
    pytest.param(
        wan22,
        wan22.WanVAE,
        "encoder",
        "conv1",
        "encode",
        [
            (torch.zeros(1, 1, 9, 2, 2, dtype=torch.float64), [_chunk(10), _chunk(20, frames=2), _chunk(30)]),
            (torch.zeros(1, 1, 1, 2, 2, dtype=torch.float64), [_chunk(40)]),
        ],
        _encoded_output,
        id="wan22-encode",
    ),
    pytest.param(
        wan22,
        wan22.WanVAE,
        "decoder",
        "conv2",
        "decode",
        [
            (
                torch.zeros(1, 1, 3, 1, 1, dtype=torch.float64),
                [_chunk(10, channels=4), _chunk(20, channels=4, frames=4), _chunk(30, channels=4, frames=4)],
            ),
            (torch.zeros(1, 1, 1, 1, 1, dtype=torch.float64), [_chunk(40, channels=4)]),
        ],
        _decoded_output,
        id="wan22-decode",
    ),
]


@pytest.mark.parametrize(
    "module,vae_class,chunk_module_name,conv_name,method_name,runs,output_transform",
    _ASSEMBLY_CASES,
)
def test_wan_vae_assembles_chunks_with_one_cat(
    monkeypatch,
    module,
    vae_class,
    chunk_module_name,
    conv_name,
    method_name,
    runs,
    output_transform,
):
    real_cat = torch.cat
    cat_calls = []

    def cat(tensors, *args, **kwargs):
        tensors = tuple(tensors)
        cat_calls.append((len(tensors), args[0] if args else kwargs.get("dim", 0)))
        return real_cat(tensors, *args, **kwargs)

    monkeypatch.setattr(module.torch, "cat", cat)

    for input_tensor, chunks in runs:
        vae = _make_vae(
            vae_class,
            **{
                chunk_module_name: _ChunkModule(chunks),
                conv_name: nn.Identity(),
            },
        )
        actual = getattr(vae, method_name)(input_tensor)
        expected = output_transform(real_cat([chunk for chunk in chunks if chunk is not None], dim=2))

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.dtype == input_tensor.dtype
        assert actual.device == input_tensor.device

    assert cat_calls == [
        (len([chunk for chunk in chunks if chunk is not None]), 2)
        for _, chunks in runs
    ]


@pytest.mark.parametrize(
    "vae_class,chunk_module_name,conv_name,method_name,input_tensor",
    [
        pytest.param(wan21.WanVAE, "encoder", "conv1", "encode", torch.empty(1, 1, 0, 1, 1), id="wan21-encode"),
        pytest.param(wan22.WanVAE, "encoder", "conv1", "encode", torch.empty(1, 1, 0, 2, 2), id="wan22-encode"),
        pytest.param(wan22.WanVAE, "decoder", "conv2", "decode", torch.empty(1, 1, 0, 1, 1), id="wan22-decode"),
    ],
)
def test_wan_vae_empty_input_behavior_is_unchanged(vae_class, chunk_module_name, conv_name, method_name, input_tensor):
    vae = _make_vae(
        vae_class,
        **{
            chunk_module_name: _ChunkModule([]),
            conv_name: nn.Identity(),
        },
    )

    with pytest.raises(UnboundLocalError):
        getattr(vae, method_name)(input_tensor)
