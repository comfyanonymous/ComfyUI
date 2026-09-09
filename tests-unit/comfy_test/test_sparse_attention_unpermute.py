import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_index_select_matches_advanced_indexing(dtype):
    out = torch.arange(48, dtype=torch.float32).reshape(8, 6).to(dtype)
    inverse = torch.tensor([3, 0, 7, 2, 6, 1, 5, 4], dtype=torch.int64)

    expected = out[inverse]
    actual = torch.index_select(out, 0, inverse)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


def test_index_select_matches_advanced_indexing_gradients():
    inverse = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
    source = torch.arange(12, dtype=torch.float32).reshape(4, 3).requires_grad_()

    expected = source[inverse].square().sum()
    expected.backward()
    expected_grad = source.grad.clone()
    source.grad = None

    actual = torch.index_select(source, 0, inverse).square().sum()
    actual.backward()

    assert torch.equal(source.grad, expected_grad)


def test_index_select_restores_partial_vsa_order():
    packed = torch.tensor(
        [[10.0], [20.0], [30.0], [40.0], [50.0], [60.0]]
    )
    inverse = torch.tensor([1, 4, 0, 5], dtype=torch.int64)

    expected = packed[inverse]
    actual = torch.index_select(packed, 0, inverse)

    assert torch.equal(actual, expected)
