from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from comfy.cli_args import args

args.cpu = True
import comfy.model_prefetch as mp
import comfy.ops as ops


def test_failed_prefetch_keeps_all_faulted_modules_reachable(monkeypatch):
    first = SimpleNamespace(_v='first', _v_signature='resident')
    second = SimpleNamespace(_v='second', _v_signature='old')
    root = SimpleNamespace(modules=lambda: iter([first, second]))
    queue = [None, root]
    monkeypatch.setattr(mp, 'PREFETCH_QUEUES', [queue])
    monkeypatch.setattr(mp, 'MALLOC_GRAPHS', {})
    monkeypatch.setattr(mp, 'cleanup_malloc_graph', lambda: None)
    vbar = ops.comfy_aimdo.model_vbar
    monkeypatch.setattr(vbar, 'vbar_fault', lambda v: 'resident' if v == 'first' else 'new')
    monkeypatch.setattr(vbar, 'vbar_signature_compare', lambda a, b: a == b)
    unpin = Mock()
    monkeypatch.setattr(vbar, 'vbar_unpin', unpin)

    def fail(*a):
        raise RuntimeError('prefetch preparation failed')
    monkeypatch.setattr(ops, 'materialize_meta_param', fail)
    monkeypatch.setattr(mp, 'pin_modules', lambda modules, device, dtype: ops.cast_modules_with_vbar(modules, None, device, None, False, True))
    with pytest.raises(RuntimeError, match='prefetch preparation failed'):
        mp.prefetch_queue_pop(queue, 'cpu', root)
    mp.cleanup_prefetch_queues()
    assert [c.args[0] for c in unpin.call_args_list] == ['first', 'second']
    assert not hasattr(first, '_prefetch')
    assert not hasattr(second, '_prefetch')
    mp.cleanup_prefetch_queues()
    assert unpin.call_count == 2


def test_successful_resident_prefetch_is_cleaned_once(monkeypatch):
    child = SimpleNamespace(_v='child', _v_signature='same')
    root = SimpleNamespace(modules=lambda: iter([child]))
    queue = [None, root, None]
    monkeypatch.setattr(mp, 'PREFETCH_QUEUES', [queue])
    monkeypatch.setattr(mp, 'MALLOC_GRAPHS', {})
    monkeypatch.setattr(mp, 'cleanup_malloc_graph', lambda: None)
    vbar = ops.comfy_aimdo.model_vbar
    monkeypatch.setattr(vbar, 'vbar_fault', lambda v: 'same')
    monkeypatch.setattr(vbar, 'vbar_signature_compare', lambda a, b: a == b)
    unpin = Mock()
    monkeypatch.setattr(vbar, 'vbar_unpin', unpin)
    monkeypatch.setattr(mp, 'pin_modules', lambda modules, device, dtype: ops.cast_modules_with_vbar(modules, None, device, None, False, True))
    core = Mock()
    mp.prefetch_queue_pop(queue, 'cpu', root, core=core)
    core.assert_called_once()
    assert child._prefetch['resident']
    unpin.assert_not_called()
    mp.prefetch_queue_pop(queue, 'cpu', None)
    unpin.assert_called_once_with('child')
    mp.cleanup_prefetch_queues()
    unpin.assert_called_once()


def test_direct_cast_failure_does_not_publish_incomplete_prefetch(monkeypatch):
    child = SimpleNamespace(_v='child', _v_signature='old')
    vbar = ops.comfy_aimdo.model_vbar
    monkeypatch.setattr(vbar, 'vbar_fault', lambda v: 'new')
    monkeypatch.setattr(vbar, 'vbar_signature_compare', lambda a, b: False)

    def fail(*a):
        raise RuntimeError('preparation failed')
    monkeypatch.setattr(ops, 'materialize_meta_param', fail)
    with pytest.raises(RuntimeError, match='preparation failed'):
        ops.cast_modules_with_vbar([child], None, 'cpu', None, False)
    assert not hasattr(child, '_prefetch')


@pytest.mark.parametrize('failure_index', [0, 1, 2])
def test_partial_batch_cleanup_only_unpins_successful_faults(monkeypatch, failure_index):
    children = [SimpleNamespace(_v=i, _v_signature='same') for i in range(3)]
    root = SimpleNamespace(modules=lambda: iter(children))
    queue = [None, root]
    monkeypatch.setattr(mp, 'PREFETCH_QUEUES', [queue])
    monkeypatch.setattr(mp, 'MALLOC_GRAPHS', {})
    monkeypatch.setattr(mp, 'cleanup_malloc_graph', lambda: None)
    vbar = ops.comfy_aimdo.model_vbar
    faulted = []

    def fault(v):
        if v == failure_index:
            raise RuntimeError('fault failed')
        faulted.append(v)
        return 'same'
    monkeypatch.setattr(vbar, 'vbar_fault', fault)
    monkeypatch.setattr(vbar, 'vbar_signature_compare', lambda a, b: a == b)
    unpin = Mock()
    monkeypatch.setattr(vbar, 'vbar_unpin', unpin)
    monkeypatch.setattr(mp, 'pin_modules', lambda modules, device, dtype: ops.cast_modules_with_vbar(modules, None, device, None, False, True))
    with pytest.raises(RuntimeError, match='fault failed'):
        mp.prefetch_queue_pop(queue, 'cpu', root)
    mp.cleanup_prefetch_queues()
    assert [c.args[0] for c in unpin.call_args_list] == faulted
    assert all(not hasattr(child, '_prefetch') for child in children)
    mp.cleanup_prefetch_queues()
    assert unpin.call_count == len(faulted)


def test_pin_modules_failure_after_cast_still_cleans_batch(monkeypatch):
    child = SimpleNamespace(_v='child', _v_signature='same', weight=torch.empty(1), bias=None, _pin_state={'fast_disk': False})
    root = SimpleNamespace(modules=lambda: iter([child]))
    queue = [None, root]
    monkeypatch.setattr(mp, 'PREFETCH_QUEUES', [queue])
    monkeypatch.setattr(mp, 'MALLOC_GRAPHS', {})
    monkeypatch.setattr(mp, 'cleanup_malloc_graph', lambda: None)
    vbar = ops.comfy_aimdo.model_vbar
    monkeypatch.setattr(vbar, 'vbar_fault', lambda v: 'same')
    monkeypatch.setattr(vbar, 'vbar_signature_compare', lambda a, b: a == b)
    unpin = Mock()
    monkeypatch.setattr(vbar, 'vbar_unpin', unpin)

    def fail(*a):
        raise RuntimeError('registration pressure')
    monkeypatch.setattr(mp.comfy.model_management, 'ensure_pin_registerable', fail)
    with pytest.raises(RuntimeError, match='registration pressure'):
        mp.prefetch_queue_pop(queue, 'cpu', root)
    mp.cleanup_prefetch_queues()
    unpin.assert_called_once_with('child')
    assert not hasattr(child, '_prefetch')


def test_null_signature_preparation_failure_does_not_unpin(monkeypatch):
    child = SimpleNamespace(_v='child', _v_signature='old')
    root = SimpleNamespace(modules=lambda: iter([child]))
    queue = [None, root]
    monkeypatch.setattr(mp, 'PREFETCH_QUEUES', [queue])
    monkeypatch.setattr(mp, 'MALLOC_GRAPHS', {})
    monkeypatch.setattr(mp, 'cleanup_malloc_graph', lambda: None)
    vbar = ops.comfy_aimdo.model_vbar
    monkeypatch.setattr(vbar, 'vbar_fault', lambda v: None)
    monkeypatch.setattr(vbar, 'vbar_signature_compare', lambda a, b: False)
    unpin = Mock()
    monkeypatch.setattr(vbar, 'vbar_unpin', unpin)

    def fail(*a):
        raise RuntimeError('preparation failed')
    monkeypatch.setattr(ops, 'materialize_meta_param', fail)
    monkeypatch.setattr(mp, 'pin_modules', lambda modules, device, dtype: ops.cast_modules_with_vbar(modules, None, device, None, False, True))
    with pytest.raises(RuntimeError, match='preparation failed'):
        mp.prefetch_queue_pop(queue, 'cpu', root)
    mp.cleanup_prefetch_queues()
    assert not hasattr(child, '_prefetch')
    unpin.assert_not_called()
