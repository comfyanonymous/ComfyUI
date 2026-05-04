from utils.json_util import merge_json_recursive


def test_merge_simple_dicts():
    base = {"a": 1, "b": 2}
    update = {"b": 3, "c": 4}
    expected = {"a": 1, "b": 3, "c": 4}
    assert merge_json_recursive(base, update) == expected


def test_merge_nested_dicts():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    update = {"a": {"y": 4, "z": 5}}
    expected = {"a": {"x": 1, "y": 4, "z": 5}, "b": 3}
    assert merge_json_recursive(base, update) == expected


def test_merge_lists():
    base = {"a": [1, 2], "b": 3}
    update = {"a": [3, 4]}
    expected = {"a": [1, 2, 3, 4], "b": 3}
    assert merge_json_recursive(base, update) == expected


def test_merge_nested_lists():
    base = {"a": {"x": [1, 2]}}
    update = {"a": {"x": [3, 4]}}
    expected = {"a": {"x": [1, 2, 3, 4]}}
    assert merge_json_recursive(base, update) == expected


def test_merge_mixed_types():
    base = {"a": [1, 2], "b": {"x": 1}}
    update = {"a": [3], "b": {"y": 2}}
    expected = {"a": [1, 2, 3], "b": {"x": 1, "y": 2}}
    assert merge_json_recursive(base, update) == expected


def test_merge_overwrite_non_dict():
    base = {"a": 1}
    update = {"a": {"x": 2}}
    expected = {"a": {"x": 2}}
    assert merge_json_recursive(base, update) == expected


def test_merge_empty_dicts():
    base = {}
    update = {"a": 1}
    expected = {"a": 1}
    assert merge_json_recursive(base, update) == expected


def test_merge_none_values():
    base = {"a": None}
    update = {"a": {"x": 1}}
    expected = {"a": {"x": 1}}
    assert merge_json_recursive(base, update) == expected


def test_merge_different_types():
    base = {"a": [1, 2]}
    update = {"a": "string"}
    expected = {"a": "string"}
    assert merge_json_recursive(base, update) == expected


def test_merge_complex_nested():
    base = {"a": [1, 2], "b": {"x": [3, 4], "y": {"p": 1}}}
    update = {"a": [5], "b": {"x": [6], "y": {"q": 2}}}
    expected = {"a": [1, 2, 5], "b": {"x": [3, 4, 6], "y": {"p": 1, "q": 2}}}
    assert merge_json_recursive(base, update) == expected
