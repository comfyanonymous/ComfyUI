"""
MIT License

Copyright (c) 2020 Corentin Garcia

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations
import typing
import typing_extensions

_K = typing.TypeVar("_K")
_V = typing.TypeVar("_V", covariant=True)


class immutabledict(typing.Mapping[_K, _V]):
    """
    An immutable wrapper around dictionaries that implements
    the complete :py:class:`collections.Mapping` interface.
    It can be used as a drop-in replacement for dictionaries
    where immutability is desired.

    Note: custom version of this class made to remove __init__
    """

    dict_cls: typing.Type[typing.Dict[typing.Any, typing.Any]] = dict
    _dict: typing.Dict[_K, _V]
    _hash: typing.Optional[int]

    @classmethod
    def fromkeys(
        cls, seq: typing.Iterable[_K], value: typing.Optional[_V] = None
    ) -> "immutabledict[_K, _V]":
        return cls(dict.fromkeys(seq, value))

    def __new__(cls, *args: typing.Any) -> typing_extensions.Self:
        inst = super().__new__(cls)
        setattr(inst, '_dict', cls.dict_cls(*args))
        setattr(inst, '_hash', None)
        return inst

    def __getitem__(self, key: _K) -> _V:
        return self._dict[key]

    def __contains__(self, key: object) -> bool:
        return key in self._dict

    def __iter__(self) -> typing.Iterator[_K]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> str:
        return "%s(%r)" % (self.__class__.__name__, self._dict)

    def __hash__(self) -> int:
        if self._hash is None:
            h = 0
            for key, value in self.items():
                h ^= hash((key, value))
            self._hash = h

        return self._hash

    def __or__(self, other: typing.Any) -> immutabledict[_K, _V]:
        if not isinstance(other, (dict, self.__class__)):
            return NotImplemented
        new = dict(self)
        new.update(other)
        return self.__class__(new)

    def __ror__(self, other: typing.Any) -> typing.Dict[typing.Any, typing.Any]:
        if not isinstance(other, (dict, self.__class__)):
            return NotImplemented
        new = dict(other)
        new.update(self)
        return new

    def __ior__(self, other: typing.Any) -> immutabledict[_K, _V]:
        raise TypeError(f"'{self.__class__.__name__}' object is not mutable")
