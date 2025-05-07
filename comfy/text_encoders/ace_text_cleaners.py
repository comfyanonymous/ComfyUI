# basic text cleaners for the ACE step model
# I didn't copy the ones from the reference code because I didn't want to deal with the dependencies
# TODO: more languages than english?

import re

def number_to_text(num, ordinal=False):
    """
    Convert a number (int or float) to its text representation.

    Args:
        num: The number to convert

    Returns:
        str: Text representation of the number
    """

    if not isinstance(num, (int, float)):
        return "Input must be a number"

    # Handle special case of zero
    if num == 0:
        return "zero"

    # Handle negative numbers
    negative = num < 0
    num = abs(num)

    # Handle floats
    if isinstance(num, float):
        # Split into integer and decimal parts
        int_part = int(num)

        # Convert both parts
        int_text = _int_to_text(int_part)

        # Handle decimal part (convert to string and remove '0.')
        decimal_str = str(num).split('.')[1]
        decimal_text = " point " + " ".join(_digit_to_text(int(digit)) for digit in decimal_str)

        result = int_text + decimal_text
    else:
        # Handle integers
        result = _int_to_text(num)

    # Add 'negative' prefix for negative numbers
    if negative:
        result = "negative " + result

    return result


def _int_to_text(num):
    """Helper function to convert an integer to text"""

    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen"]

    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    if num < 20:
        return ones[num]

    if num < 100:
        return tens[num // 10] + (" " + ones[num % 10] if num % 10 != 0 else "")

    if num < 1000:
        return ones[num // 100] + " hundred" + (" " + _int_to_text(num % 100) if num % 100 != 0 else "")

    if num < 1000000:
        return _int_to_text(num // 1000) + " thousand" + (" " + _int_to_text(num % 1000) if num % 1000 != 0 else "")

    if num < 1000000000:
        return _int_to_text(num // 1000000) + " million" + (" " + _int_to_text(num % 1000000) if num % 1000000 != 0 else "")

    return _int_to_text(num // 1000000000) + " billion" + (" " + _int_to_text(num % 1000000000) if num % 1000000000 != 0 else "")


def _digit_to_text(digit):
    """Convert a single digit to text"""
    digits = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    return digits[digit]


_whitespace_re = re.compile(r"\s+")


# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = {
    "en": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("mrs", "misess"),
            ("mr", "mister"),
            ("dr", "doctor"),
            ("st", "saint"),
            ("co", "company"),
            ("jr", "junior"),
            ("maj", "major"),
            ("gen", "general"),
            ("drs", "doctors"),
            ("rev", "reverend"),
            ("lt", "lieutenant"),
            ("hon", "honorable"),
            ("sgt", "sergeant"),
            ("capt", "captain"),
            ("esq", "esquire"),
            ("ltd", "limited"),
            ("col", "colonel"),
            ("ft", "fort"),
        ]
    ],
}


def expand_abbreviations_multilingual(text, lang="en"):
    for regex, replacement in _abbreviations[lang]:
        text = re.sub(regex, replacement, text)
    return text


_symbols_multilingual = {
    "en": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " and "),
            ("@", " at "),
            ("%", " percent "),
            ("#", " hash "),
            ("$", " dollar "),
            ("£", " pound "),
            ("°", " degree "),
        ]
    ],
}


def expand_symbols_multilingual(text, lang="en"):
    for regex, replacement in _symbols_multilingual[lang]:
        text = re.sub(regex, replacement, text)
        text = text.replace("  ", " ")  # Ensure there are no double spaces
    return text.strip()


_ordinal_re = {
    "en": re.compile(r"([0-9]+)(st|nd|rd|th)"),
}
_number_re = re.compile(r"[0-9]+")
_currency_re = {
    "USD": re.compile(r"((\$[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+\$))"),
    "GBP": re.compile(r"((£[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+£))"),
    "EUR": re.compile(r"(([0-9\.\,]*[0-9]+€)|((€[0-9\.\,]*[0-9]+)))"),
}

_comma_number_re = re.compile(r"\b\d{1,3}(,\d{3})*(\.\d+)?\b")
_dot_number_re = re.compile(r"\b\d{1,3}(.\d{3})*(\,\d+)?\b")
_decimal_number_re = re.compile(r"([0-9]+[.,][0-9]+)")


def _remove_commas(m):
    text = m.group(0)
    if "," in text:
        text = text.replace(",", "")
    return text


def _remove_dots(m):
    text = m.group(0)
    if "." in text:
        text = text.replace(".", "")
    return text


def _expand_decimal_point(m, lang="en"):
    amount = m.group(1).replace(",", ".")
    return number_to_text(float(amount))


def _expand_currency(m, lang="en", currency="USD"):
    amount = float((re.sub(r"[^\d.]", "", m.group(0).replace(",", "."))))
    full_amount = number_to_text(amount)

    and_equivalents = {
        "en": ", ",
        "es": " con ",
        "fr": " et ",
        "de": " und ",
        "pt": " e ",
        "it": " e ",
        "pl": ", ",
        "cs": ", ",
        "ru": ", ",
        "nl": ", ",
        "ar": ", ",
        "tr": ", ",
        "hu": ", ",
        "ko": ", ",
    }

    if amount.is_integer():
        last_and = full_amount.rfind(and_equivalents[lang])
        if last_and != -1:
            full_amount = full_amount[:last_and]

    return full_amount


def _expand_ordinal(m, lang="en"):
    return number_to_text(int(m.group(1)), ordinal=True)


def _expand_number(m, lang="en"):
    return number_to_text(int(m.group(0)))


def expand_numbers_multilingual(text, lang="en"):
    if lang in ["en", "ru"]:
        text = re.sub(_comma_number_re, _remove_commas, text)
    else:
        text = re.sub(_dot_number_re, _remove_dots, text)
    try:
        text = re.sub(_currency_re["GBP"], lambda m: _expand_currency(m, lang, "GBP"), text)
        text = re.sub(_currency_re["USD"], lambda m: _expand_currency(m, lang, "USD"), text)
        text = re.sub(_currency_re["EUR"], lambda m: _expand_currency(m, lang, "EUR"), text)
    except:
        pass

    text = re.sub(_decimal_number_re, lambda m: _expand_decimal_point(m, lang), text)
    text = re.sub(_ordinal_re[lang], lambda m: _expand_ordinal(m, lang), text)
    text = re.sub(_number_re, lambda m: _expand_number(m, lang), text)
    return text


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def multilingual_cleaners(text, lang):
    text = text.replace('"', "")
    if lang == "tr":
        text = text.replace("İ", "i")
        text = text.replace("Ö", "ö")
        text = text.replace("Ü", "ü")
    text = lowercase(text)
    try:
        text = expand_numbers_multilingual(text, lang)
    except:
        pass
    try:
        text = expand_abbreviations_multilingual(text, lang)
    except:
        pass
    try:
        text = expand_symbols_multilingual(text, lang=lang)
    except:
        pass
    text = collapse_whitespace(text)
    return text


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text
