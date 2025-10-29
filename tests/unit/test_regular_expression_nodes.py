import re

import pytest

from comfy_extras.nodes.nodes_regexp import RegexFlags, Regex, RegexMatchGroupByIndex, RegexMatchGroupByName, \
    RegexMatchExpand


def test_regex_flags():
    n = RegexFlags()

    # Test with no flags
    flags, = n.execute(ASCII=False, IGNORECASE=False, LOCALE=False, MULTILINE=False,
                       DOTALL=False, VERBOSE=False, UNICODE=False, NOFLAG=True)
    assert flags == 0

    # Test single flag
    flags, = n.execute(ASCII=True, IGNORECASE=False, LOCALE=False, MULTILINE=False,
                       DOTALL=False, VERBOSE=False, UNICODE=False, NOFLAG=False)
    assert flags == re.ASCII

    # Test multiple flags
    flags, = n.execute(ASCII=True, IGNORECASE=True, LOCALE=False, MULTILINE=False,
                       DOTALL=False, VERBOSE=False, UNICODE=False, NOFLAG=False)
    assert flags == (re.ASCII | re.IGNORECASE)


def test_regex():
    n = Regex()

    # Basic match test
    match, *_ = n.execute(pattern=r"hello", string="hello world")
    assert match is not None
    assert match.group(0) == "hello"

    # Test with flags
    match, *_ = n.execute(pattern=r"HELLO", string="hello world", flags=re.IGNORECASE)
    assert match is not None
    assert match.group(0) == "hello"

    # Test no match
    match, has_match = n.execute(pattern=r"python", string="hello world")
    assert match is None
    assert not has_match


def test_regex_match_group_by_index():
    n = RegexMatchGroupByIndex()
    regex = Regex()

    # Test basic group
    match, *_ = regex.execute(pattern=r"(hello) (world)", string="hello world")
    group, = n.execute(match=match, index=0)
    assert group == "hello world"

    group, = n.execute(match=match, index=1)
    assert group == "hello"

    group, = n.execute(match=match, index=2)
    assert group == "world"


def test_regex_match_group_by_name():
    n = RegexMatchGroupByName()
    regex = Regex()

    # Test named group
    match, *_ = regex.execute(pattern=r"(?P<greeting>hello) (?P<subject>world)",
                              string="hello world")

    group, = n.execute(match=match, name="greeting")
    assert group == "hello"

    group, = n.execute(match=match, name="subject")
    assert group == "world"

    # Test non-existent group name
    with pytest.raises(IndexError):
        n.execute(match=match, name="nonexistent")


def test_regex_match_expand():
    n = RegexMatchExpand()
    regex = Regex()

    # Test basic expansion
    match, *_ = regex.execute(pattern=r"(hello) (world)", string="hello world")
    result, = n.execute(match=match, template=r"\2, \1!")
    assert result == "world, hello!"

    # Test named group expansion
    match, *_ = regex.execute(pattern=r"(?P<greeting>hello) (?P<subject>world)",
                              string="hello world")
    result, = n.execute(match=match, template=r"\g<subject>, \g<greeting>!")
    assert result == "world, hello!"
