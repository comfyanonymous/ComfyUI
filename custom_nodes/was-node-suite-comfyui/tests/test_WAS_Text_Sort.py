from was_mock import was_text_sort

def test_empty_text():
    assert was_text_sort() == ""

def test_empty_text_with_separator_override():
    assert was_text_sort(separator="|") == ""

def test_already_sorted_text():
    assert was_text_sort("already, sorted, text") == "already, sorted, text"

def test_already_sorted_text_with_separator_override():
    assert was_text_sort("already, sorted, text", separator="|") == "already, sorted, text"

def test_with_alternative_separator():
    assert was_text_sort("test | with | alternative | separator", separator=" | ") == "alternative | separator | test | with"

def test_with_trailing_separators():
    assert was_text_sort("test, with, trailing, separator,") == "separator, test, trailing, with"

def test_with_tabs():
    assert was_text_sort("test,\t without, \tweights") == "test, weights, without"

def test_with_linefeed_newlines():
    assert was_text_sort("test,\n without, \nweights") == "test, weights, without"

def test_with_macos_pre_cheetah_newlines():
    assert was_text_sort("test,\r without, \rweights") == "test, weights, without"

def test_with_windows_newlines():
    assert was_text_sort("test,\r\n without, \r\nweights") == "test, weights, without"

def test_without_weights():
    assert was_text_sort("test, without, weights") == "test, weights, without"

def test_with_weights():
    assert was_text_sort("(test:1), (with:2.0), (weights:3.1)") == "(test:1), (weights:3.1), (with:2.0)"

def test_with_some_weights():
    assert was_text_sort("(test:1), with, some, (weights:3.1)") == "some, (test:1), (weights:3.1), with"

def test_with_half_weights():
    assert was_text_sort("(test:1), with, half (weights:3.1)") == "half (weights:3.1), (test:1), with"

# ASCII "_" is after uppercase and before lowercase letters
def test_with_wildcards():
    assert was_text_sort("test, with, __wildcards__") == "__wildcards__, test, with"

def test_with_weighted_wildcards():
    assert was_text_sort("test, (with:2), (__wildcards__:3)") == "(__wildcards__:3), test, (with:2)"

# ASCII "{" is after all letters
def test_with_dynamic_prompts():
    assert was_text_sort("test, {with|dynamic|prompts}") == "test, {with|dynamic|prompts}"

def test_with_weighted_dynamic_prompts():
    assert was_text_sort("(test:1.1), with, ({weighted|dynamic|prompts}:0.9)") == "(test:1.1), with, ({weighted|dynamic|prompts}:0.9)"

def test_with_embeddings():
    assert was_text_sort("test, with, embedding:my_embed.pt") == "embedding:my_embed.pt, test, with"

def test_with_lora():
    assert was_text_sort("test, with, lora:my_lora.safetensors") == "lora:my_lora.safetensors, test, with"

def test_with_grouped_weights():
    assert was_text_sort("(test, with:1), (grouped, weights:2.1)") == "(grouped, weights:2.1), (test, with:1)"

def test_with_nested_weights():
    assert was_text_sort("(test, (with:1.2):1.1), ((nested:1), weights:2)") == "((nested:1), weights:2), (test, (with:1.2):1.1)"