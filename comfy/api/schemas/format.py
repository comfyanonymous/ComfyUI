import datetime
import decimal
import functools
import typing
import uuid

from dateutil import parser, tz


class CustomIsoparser(parser.isoparser):
    def __init__(self, sep: typing.Optional[str] = None):
        """
        :param sep:
            A single character that separates date and time portions. If
            ``None``, the parser will accept any single character.
            For strict ISO-8601 adherence, pass ``'T'``.
        """
        if sep is not None:
            if (len(sep) != 1 or ord(sep) >= 128 or sep in '0123456789'):
                raise ValueError('Separator must be a single, non-numeric ' +
                                 'ASCII character')

            used_sep = sep.encode('ascii')
        else:
            used_sep = None

        self._sep = used_sep

    @staticmethod
    def __get_ascii_bytes(str_in: str) -> bytes:
        # If it's unicode, turn it into bytes, since ISO-8601 only covers ASCII
        # ASCII is the same in UTF-8
        try:
            return str_in.encode('ascii')
        except UnicodeEncodeError as e:
            msg = 'ISO-8601 strings should contain only ASCII characters'
            raise ValueError(msg) from e

    def __parse_isodate(self, dt_str: str) -> typing.Tuple[typing.Tuple[int, int, int], int]:
        dt_str_ascii = self.__get_ascii_bytes(dt_str)
        values = self._parse_isodate(dt_str_ascii) # type: ignore
        values = typing.cast(typing.Tuple[typing.List[int], int], values)
        components = typing.cast( typing.Tuple[int, int, int], tuple(values[0]))
        pos = values[1]
        return components, pos

    def __parse_isotime(self, dt_str: str) -> typing.Tuple[int, int, int, int, typing.Optional[typing.Union[tz.tzutc, tz.tzoffset]]]:
        dt_str_ascii = self.__get_ascii_bytes(dt_str)
        values = self._parse_isotime(dt_str_ascii) # type: ignore
        components: typing.Tuple[int, int, int, int, typing.Optional[typing.Union[tz.tzutc, tz.tzoffset]]] = tuple(values) # type: ignore
        return components

    def parse_isodatetime(self, dt_str: str) -> datetime.datetime:
        date_components, pos = self.__parse_isodate(dt_str)
        if len(dt_str) <= pos:
            # len(components) <= 3
            raise ValueError('Value is not a datetime')
        if self._sep is None or dt_str[pos:pos + 1] == self._sep:
            hour, minute, second, microsecond, tzinfo = self.__parse_isotime(dt_str[pos + 1:])
            if hour == 24:
                hour = 0
                components = (*date_components, hour, minute, second, microsecond, tzinfo)
                return datetime.datetime(*components) + datetime.timedelta(days=1)
            else:
                components = (*date_components, hour, minute, second, microsecond, tzinfo)
        else:
            raise ValueError('String contains unknown ISO components')

        return datetime.datetime(*components)

    def parse_isodate_str(self, datestr: str) -> datetime.date:
        components, pos = self.__parse_isodate(datestr)

        if len(datestr) > pos:
            raise ValueError('String contains invalid time components')

        if len(components) > 3:
            raise ValueError('String contains invalid time components')

        return datetime.date(*components)

DEFAULT_ISOPARSER = CustomIsoparser()

@functools.lru_cache()
def as_date(arg: str) -> datetime.date:
    """
    type = "string"
    format = "date"
    """
    return DEFAULT_ISOPARSER.parse_isodate_str(arg)

@functools.lru_cache()
def as_datetime(arg: str) -> datetime.datetime:
    """
    type = "string"
    format = "date-time"
    """
    return DEFAULT_ISOPARSER.parse_isodatetime(arg)

@functools.lru_cache()
def as_decimal(arg: str) -> decimal.Decimal:
    """
    Applicable when storing decimals that are sent over the wire as strings
    type = "string"
    format = "number"
    """
    return decimal.Decimal(arg)

@functools.lru_cache()
def as_uuid(arg: str) -> uuid.UUID:
    """
    type = "string"
    format = "uuid"
    """
    return uuid.UUID(arg)