import re
import random

# This function is used to strip comments from a string.
# It supports both single-line and multi-line comments of the following formats:
# // Single-line comment
# /* Multi-line comment */
#
# @param text The string to strip comments from.
# @return The string with comments stripped.
def strip_comments(text):
    return re.sub(r'/\*.*?\*/|//.*?$', '', text, flags=re.MULTILINE|re.DOTALL)

# This function is used to generate a dynamic prompt from a string.
# It allows you to include random text in your prompt, contained within option blocks of the following format:
# {option1|option2|option3}
#
# For example, the following prompt:
# > "This is a {good|great|spectacular} prompt."
#
# Could generate any of the following prompts:
# > "This is a good prompt."
# > "This is a great prompt."
# > "This is a spectacular prompt."
#
# If you need to include a '{' or '}' character in your prompt, you can escape it by doubling it, like so:
# > "This is a {{ and }} character."
#
# Which would generate the following prompt:
# > "This is a { and } character."
#
# It also strips comments from the prompt, so you can include comments in your prompt like so:
#
# @param text The string to generate a dynamic prompt from.
# @return The generated dynamic prompt.
def process_dynamic_prompt(text):
    prompt = strip_comments(text)

    new_prompt = ""
    options = []
    inside_option = False

    i = 0
    while i < len(prompt):
        char = prompt[i]
        if char == "{":
            if inside_option:
                if i < len(prompt) - 1 and prompt[i + 1] == "{":
                    options[-1] += char
                    i += 2
                    continue
                else:
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
                raise ValueError(f"Option block closed at position {i} without being opened. "
                                 "Make sure each '{' has a matching '}'. "
                                 "If you want to include a '}' without closing an option block, use '}}'.")
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
        raise ValueError("Option block opened but not closed. "
                         "Make sure each '{' has a matching '}'. "
                         "If you want to include a '}' without closing an option block, use '}}'.")

    return new_prompt

# Checks if the given string is a valid dynamic prompt.
# @param text The string to check.
# @return An error message if the string is not a valid dynamic prompt, True otherwise.
def validate_dynamic_prompt(text):
    try:
        process_dynamic_prompt(text)
    except ValueError as e:
        print("validate_dynamic_prompt: %s" % str(e))
        return str(e)
    
    print("validate_dynamic_prompt: True")
    return True

# Checks if the given string contains any dynamic prompts.
# @param text The string to check.
# @return True if the string contains any dynamic prompts, False otherwise.
def contains_dynamic_prompt(text):
    # This should check if there are any { or } characters that aren't escaped
    return re.search(r'(?<!\{)\{(?!\{)|(?<!\})\}(?!\})', text) is not None


# Checks a prompt and returns a hash of it if there are no dynamic elements, or NaN if there are, thus forcing it to always change.
# @param prompt The prompt to check.
# @return A hash of the prompt if there are no dynamic elements, or NaN if there are.
def is_dynamic_prompt_changed(prompt):
    if contains_dynamic_prompt(prompt):
        print("is_dynamic_prompt_changed: NaN")
        return float("nan")
    else:
        print("is_dynamic_prompt_changed: %s" % hash(prompt))
        return prompt
