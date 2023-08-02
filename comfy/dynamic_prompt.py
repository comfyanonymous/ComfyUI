"""Contains functions for processing dynamic prompts."""
import re
import random
from typing import List, Literal, Union


def strip_comments(text: str) -> str:
    """
    Strips comments from a string.
    Supports both single-line and multi-line comments of the following formats:
    // Single-line comment
    /* Multi-line comment */

    Args:
        text (str): The string to strip comments from.

    Returns:
        str: The string with comments stripped.
    """
    return re.sub(r'/\*.*?\*/|//.*?$', '', text, flags=re.MULTILINE | re.DOTALL)


def process_dynamic_prompt(text: str) -> str:
    """
    Generates a dynamic prompt from a string.
    Allows you to include random text in your prompt, contained within option
    blocks of the following format:
    {option1|option2|option3}

    For example, the following prompt:
    > "This is a {good|great|spectacular} prompt."

    Could generate any of the following prompts:
    > "This is a good prompt."
    > "This is a great prompt."
    > "This is a spectacular prompt."

    If you need to include a '{' or '}' character in your prompt, you can escape
    it by doubling it, like so:
    > "This is a {{ and }} character."

    Which would generate the following prompt:
    > "This is a { and } character."

    It also strips comments from the prompt, so you can include comments in your prompt like so:

    Args:
        text (str): The string to generate a dynamic prompt from.

    Returns:
        str: The generated dynamic prompt.
    """
    prompt = strip_comments(text)

    new_prompt = ""
    options: List[str] = []
    inside_option = False

    i: int = 0
    while i < len(prompt):
        char = prompt[i]
        if char == "{":
            if inside_option:
                if i < len(prompt) - 1 and prompt[i + 1] == "{":
                    options[-1] += char
                    i += 2
                    continue
                raise ValueError(f"Option block opened at position {i} inside another option block. "
                                 "Each option block should be closed before starting a new one. "
                                 "If you want to include a '{' inside an option block, use '{{'.")
            else:
                if i < len(prompt) - 1 and prompt[i + 1] == "{":
                    new_prompt += char
                    i += 2
                    continue
                else:
                    inside_option = True
                    options.append("")
        elif char == "}":
            if not inside_option or (i < len(prompt) - 1 and prompt[i + 1] == "}"):
                if inside_option:
                    random_option = random.choice(options[-1].split("|"))
                    new_prompt += random_option
                else:
                    new_prompt += char
                i += 2
                continue
            if not inside_option:
                raise ValueError(
                    f"Option block closed at position {i} without being opened. "
                    "Make sure each '{' has a matching '}'. "
                    "If you want to include a '}' without closing an option block, use '}}'."
                )
            inside_option = False
            random_option = random.choice(options[-1].split("|"))
            new_prompt += random_option
        else:
            if inside_option:
                options[-1] += char
            else:
                new_prompt += char
        i += 1

    if inside_option:
        raise ValueError(
            "Option block opened but not closed. "
            "Make sure each '{' has a matching '}'. "
            "If you want to include a '}' without closing an option block, use '}}'."
        )

    return new_prompt


def validate_dynamic_prompt(text: str) -> Union[Literal[True], str]:
    """
    Checks if the given string is a valid dynamic prompt.

    Args:
        text (str): The string to check.

    Returns:
        str: An error message if the string is not a valid dynamic prompt, True otherwise.
    """
    try:
        process_dynamic_prompt(text)
    except ValueError as err:
        print(f"validate_dynamic_prompt: {err}")
        return str(err)

    print("validate_dynamic_prompt: True")
    return True


def contains_dynamic_prompt(text: str) -> bool:
    """
    Checks if the given string contains any dynamic prompts.

    Args:
        text (str): The string to check.

    Returns:
        bool: True if the string contains any dynamic prompts, False otherwise.
    """
    # This should check if there are any { or } characters that aren't escaped
    return re.search(r'(?<!\{)\{(?!\{)|(?<!\})\}(?!\})', text) is not None


def is_dynamic_prompt_changed(prompt: str) -> Union[float, str]:
    """
    Checks a prompt and returns it if there are no dynamic elements,
    or NaN if there are, thus forcing it to always change.

    Args:
        prompt (str): The prompt to check.

    Returns:
        Union[float, str]: A hash of the prompt if there are no dynamic elements,
          or NaN if there are.
    """
    if contains_dynamic_prompt(prompt):
        return float("nan")

    return prompt
