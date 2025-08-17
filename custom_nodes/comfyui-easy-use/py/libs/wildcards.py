import json
import os
import random
import re
from math import prod

import yaml

import folder_paths

from .log import log_node_info

easy_wildcard_dict = {}

def get_wildcard_list():
    return [f"__{x}__" for x in easy_wildcard_dict.keys()]

def wildcard_normalize(x):
    return x.replace("\\", "/").lower()

def read_wildcard(k, v):
    if isinstance(v, list):
        k = wildcard_normalize(k)
        easy_wildcard_dict[k] = v
    elif isinstance(v, dict):
        for k2, v2 in v.items():
            new_key = f"{k}/{k2}"
            new_key = wildcard_normalize(new_key)
            read_wildcard(new_key, v2)

def read_wildcard_dict(wildcard_path):
    global easy_wildcard_dict
    for root, directories, files in os.walk(wildcard_path, followlinks=True):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, wildcard_path)
                key = os.path.splitext(rel_path)[0].replace('\\', '/').lower()

                try:
                    with open(file_path, 'r', encoding="UTF-8", errors="ignore") as f:
                        lines = f.read().splitlines()
                        easy_wildcard_dict[key] = lines
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding="ISO-8859-1") as f:
                        lines = f.read().splitlines()
                        easy_wildcard_dict[key] = lines
            elif file.endswith('.yaml'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    yaml_data = yaml.load(f, Loader=yaml.FullLoader)

                    for k, v in yaml_data.items():
                        read_wildcard(k, v)
            elif file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        json_data = json.load(f)
                        for key, value in json_data.items():
                            key = wildcard_normalize(key)
                            easy_wildcard_dict[key] = value
                except ValueError:
                    print('json files load error')
    return easy_wildcard_dict


def process(text, seed=None):

    if seed is not None:
        random.seed(seed)

    def replace_options(string):
        replacements_found = False

        def replace_option(match):
            nonlocal replacements_found
            options = match.group(1).split('|')

            multi_select_pattern = options[0].split('$$')
            select_range = None
            select_sep = ' '
            range_pattern = r'(\d+)(-(\d+))?'
            range_pattern2 = r'-(\d+)'

            if len(multi_select_pattern) > 1:
                r = re.match(range_pattern, options[0])

                if r is None:
                    r = re.match(range_pattern2, options[0])
                    a = '1'
                    b = r.group(1).strip()
                else:
                    a = r.group(1).strip()
                    b = r.group(3).strip()

                if r is not None:
                    if b is not None and is_numeric_string(a) and is_numeric_string(b):
                        # PATTERN: num1-num2
                        select_range = int(a), int(b)
                    elif is_numeric_string(a):
                        # PATTERN: num
                        x = int(a)
                        select_range = (x, x)

                    if select_range is not None and len(multi_select_pattern) == 2:
                        # PATTERN: count$$
                        options[0] = multi_select_pattern[1]
                    elif select_range is not None and len(multi_select_pattern) == 3:
                        # PATTERN: count$$ sep $$
                        select_sep = multi_select_pattern[1]
                        options[0] = multi_select_pattern[2]

            adjusted_probabilities = []

            total_prob = 0

            for option in options:
                parts = option.split('::', 1)
                if len(parts) == 2 and is_numeric_string(parts[0].strip()):
                    config_value = float(parts[0].strip())
                else:
                    config_value = 1  # Default value if no configuration is provided

                adjusted_probabilities.append(config_value)
                total_prob += config_value

            normalized_probabilities = [prob / total_prob for prob in adjusted_probabilities]

            if select_range is None:
                select_count = 1
            else:
                select_count = random.randint(select_range[0], select_range[1])

            if select_count > len(options):
                selected_items = options
            else:
                selected_items = random.choices(options, weights=normalized_probabilities, k=select_count)
                selected_items = set(selected_items)

                try_count = 0
                while len(selected_items) < select_count and try_count < 10:
                    remaining_count = select_count - len(selected_items)
                    additional_items = random.choices(options, weights=normalized_probabilities, k=remaining_count)
                    selected_items |= set(additional_items)
                    try_count += 1

            selected_items2 = [re.sub(r'^\s*[0-9.]+::', '', x, 1) for x in selected_items]
            replacement = select_sep.join(selected_items2)
            if '::' in replacement:
                pass

            replacements_found = True
            return replacement

        pattern = r'{([^{}]*?)}'
        replaced_string = re.sub(pattern, replace_option, string)

        return replaced_string, replacements_found

    def replace_wildcard(string):
        global easy_wildcard_dict
        pattern = r"__([\w\s.\-+/*\\]+?)__"
        matches = re.findall(pattern, string)
        replacements_found = False

        for match in matches:
            keyword = match.lower()
            keyword = wildcard_normalize(keyword)
            if keyword in easy_wildcard_dict:
                replacement = random.choice(easy_wildcard_dict[keyword])
                replacements_found = True
                string = string.replace(f"__{match}__", replacement, 1)
            elif '*' in keyword:
                subpattern = keyword.replace('*', '.*').replace('+', r'\+')
                total_patterns = []
                found = False
                for k, v in easy_wildcard_dict.items():
                    if re.match(subpattern, k) is not None:
                        total_patterns += v
                        found = True

                if found:
                    replacement = random.choice(total_patterns)
                    replacements_found = True
                    string = string.replace(f"__{match}__", replacement, 1)
            elif '/' not in keyword:
                string_fallback = string.replace(f"__{match}__", f"__*/{match}__", 1)
                string, replacements_found = replace_wildcard(string_fallback)

        return string, replacements_found

    replace_depth = 100
    stop_unwrap = False
    while not stop_unwrap and replace_depth > 1:
        replace_depth -= 1  # prevent infinite loop

        # pass1: replace options
        pass1, is_replaced1 = replace_options(text)

        while is_replaced1:
            pass1, is_replaced1 = replace_options(pass1)

        # pass2: replace wildcards
        text, is_replaced2 = replace_wildcard(pass1)
        stop_unwrap = not is_replaced1 and not is_replaced2

    return text


def is_numeric_string(input_str):
    return re.match(r'^-?\d+(\.\d+)?$', input_str) is not None


def safe_float(x):
    if is_numeric_string(x):
        return float(x)
    else:
        return 1.0


def extract_lora_values(string):
    pattern = r'<lora:([^>]+)>'
    matches = re.findall(pattern, string)

    def touch_lbw(text):
        return re.sub(r'LBW=[A-Za-z][A-Za-z0-9_-]*:', r'LBW=', text)

    items = [touch_lbw(match.strip(':')) for match in matches]

    added = set()
    result = []
    for item in items:
        item = item.split(':')

        lora = None
        a = None
        b = None
        lbw = None
        lbw_a = None
        lbw_b = None

        if len(item) > 0:
            lora = item[0]

            for sub_item in item[1:]:
                if is_numeric_string(sub_item):
                    if a is None:
                        a = float(sub_item)
                    elif b is None:
                        b = float(sub_item)
                elif sub_item.startswith("LBW="):
                    for lbw_item in sub_item[4:].split(';'):
                        if lbw_item.startswith("A="):
                            lbw_a = safe_float(lbw_item[2:].strip())
                        elif lbw_item.startswith("B="):
                            lbw_b = safe_float(lbw_item[2:].strip())
                        elif lbw_item.strip() != '':
                            lbw = lbw_item

        if a is None:
            a = 1.0
        if b is None:
            b = 1.0

        if lora is not None and lora not in added:
            result.append((lora, a, b, lbw, lbw_a, lbw_b))
            added.add(lora)

    return result


def remove_lora_tags(string):
    pattern = r'<lora:[^>]+>'
    result = re.sub(pattern, '', string)

    return result

def process_with_loras(wildcard_opt, model, clip, title="Positive", seed=None, can_load_lora=True, pipe_lora_stack=[], easyCache=None):
    pass1 = process(wildcard_opt, seed)
    loras = extract_lora_values(pass1)
    pass2 = remove_lora_tags(pass1)

    has_noodle_key = True if "__" in wildcard_opt else False
    has_loras = True if loras != [] else False
    show_wildcard_prompt = True if has_noodle_key or has_loras else False

    if can_load_lora and has_loras:
        for lora_name, model_weight, clip_weight, lbw, lbw_a, lbw_b in loras:
            if (lora_name.split('.')[-1]) not in folder_paths.supported_pt_extensions:
                lora_name = lora_name+".safetensors"
            lora = {
                "lora_name": lora_name, "model": model, "clip": clip, "model_strength": model_weight,
                "clip_strength": clip_weight,
                "lbw_a": lbw_a,
                "lbw_b": lbw_b,
                "lbw": lbw
            }
            model, clip = easyCache.load_lora(lora)
            lora["model"] = model
            lora["clip"] = clip
            pipe_lora_stack.append(lora)

    log_node_info("easy wildcards",f"{title}: {pass2}")
    if pass1 != pass2:
        log_node_info("easy wildcards",f'{title}_decode: {pass1}')

    return model, clip, pass2, pass1, show_wildcard_prompt, pipe_lora_stack


def expand_wildcard(keyword: str) -> tuple[str]:
    """传入文件通配符的关键词，从 easy_wildcard_dict 中获取通配符的所有选项。"""
    global easy_wildcard_dict
    if keyword in easy_wildcard_dict:
        return tuple(easy_wildcard_dict[keyword])
    elif '*' in keyword:
        subpattern = keyword.replace('*', '.*').replace('+', r"\+")
        total_pattern = []
        for k, v in easy_wildcard_dict.items():
            if re.match(subpattern, k) is not None:
                total_pattern.extend(v)
        if total_pattern:
            return tuple(total_pattern)
    elif '/' not in keyword:
        return expand_wildcard(f"*/{keyword}")

def expand_options(options: str) -> tuple[str]:
    """传入去掉 {} 的选项。
    展开选项通配符，返回该选项中的每一项，这里的每一项都是一个替换项。
    不会对选项内容进行任何处理，即便存在空格或特殊符号，也会原样返回。"""
    return tuple(options.split("|"))


def decimal_to_irregular(n, bases):
    """
    将十进制数转换为不规则进制

    :param n: 十进制数
    :param bases: 各位置的基数列表，从低位到高位
    :return: 不规则进制表示的列表，从低位到高位
    """
    if n == 0:
        return [0] * len(bases) if bases else [0]

    digits = []
    remaining = n

    # 从低位到高位处理
    for base in bases:
        digit = remaining % base
        digits.append(digit)
        remaining = remaining // base

    return digits


class WildcardProcessor:
    """通配符处理器

    通配符格式：
    + option  :   {a|b}
    + wildcard:   __keyword__ 通配符内容将从 Easy-Use 插件提供的 easy_wildcard_dict 中获取
    """

    RE_OPTIONS = re.compile(r"{([^{}]*?)}")
    RE_WILDCARD = re.compile(r"__([\w\s.\-+/*\\]+?)__")
    RE_REPLACER = re.compile(r"{([^{}]*?)}|__([\w\s.\-+/*\\]+?)__")

    # 将输入的提示词转化成符合 python str.format 要求格式的模板，并将 option 和 wildcard 按照顺序在模板中留下 {0}, {1} 等占位符
    template: str
    # option、wildcard 的替换项列表，按照在模板中出现的顺序排列，相同的替换项列表只保留第一份
    replacers: dict[int, tuple[str]]
    # 占位符的编号和替换项列表的索引的映射，占位符编号按照在模板中出现的顺序排列，方便减少替换项的存储占用
    placeholder_mapping: dict[str, int]  # placeholder_id => replacer_id
    # 各替换项列表的项数，按照在模板中出现的顺序排列，提前计算，方便后续使用
    placeholder_choices: dict[str, int]  # placeholder_id => len(replacer)

    def __init__(self, text: str):
        self.__make_template(text)
        self.__total = None

    def random(self, seed=None) -> str:
        "从所有可能性中随机获取一个"
        if seed is not None:
            random.seed(seed)
        return self.getn(random.randint(0, self.total() - 1))

    def getn(self, n: int) -> str:
        "从所有可能性中获取第 n 个，以 self.total() 为周期循环"
        n = n % self.total()
        indice = decimal_to_irregular(n, self.placeholder_choices.values())
        replacements = {
            placeholder_id: self.replacers[self.placeholder_mapping[placeholder_id]][i]
            for placeholder_id, i in zip(self.placeholder_mapping.keys(), indice)
        }
        return self.template.format(**replacements)

    def getmany(self, limit: int, offset: int = 0) -> list[str]:
        """返回一组可能性组成的列表，为了避免结果太长导致内存占用超限，使用 limit 限制列表的长度，使用 offset 调整偏移。
        若 limit 和 offset 的设置导致预期的结果长度超过剩下的实际长度，则会回到开头。
        """
        return [self.getn(n) for n in range(offset, offset + limit)]

    def total(self) -> int:
        "计算可能性的数目"
        if self.__total is None:
            self.__total = prod(self.placeholder_choices.values())
        return self.__total

    def __make_template(self, text: str):
        """将输入的提示词转化成符合 python str.format 要求格式的模板，
        并将 option 和 wildcard 按照顺序在模板中留下 {r0}, {r1} 等占位符，
        即使遇到相同的 option 或 wildcard，留下的占位符编号也不同，从而使每项都独立变化。
        """
        self.placeholder_mapping = {}
        placeholder_id = 0
        replacer_id = 0
        replacers_rev = {}  # replacers => id
        blocks = []
        # 记录所处理过的通配符末尾在文本中的位置，用于拼接完整的模板
        tail = 0
        for match in self.RE_REPLACER.finditer(text):
            # 提取并展开通配符内容
            m = match.group(0)
            if m.startswith("{"):
                choices = expand_options(m[1:-1])
            elif m.startswith("__"):
                keyword = m[2:-2].lower()
                keyword = wildcard_normalize(keyword)
                choices = expand_wildcard(keyword)
            else:
                raise ValueError(f"{m!r} is not a wildcard or option")

            # 记录通配符的替换项列表和ID，相同的通配符只保留第一个
            if choices not in replacers_rev:
                replacers_rev[choices] = replacer_id
                replacer_id += 1

            # 拼接通配符前方文本
            start, end = match.span()
            blocks.append(text[tail:start])
            tail = end
            # 将通配符替换为占位符，并记录占位符和替换项列表的索引的映射
            blocks.append(f"{{r{placeholder_id}}}")
            self.placeholder_mapping[f"r{placeholder_id}"] = replacers_rev[choices]
            placeholder_id += 1

        if tail < len(text):
            blocks.append(text[tail:])
        self.template = "".join(blocks)
        self.replacers = {v: k for k, v in replacers_rev.items()}
        self.placeholder_choices = {
            placeholder_id: len(self.replacers[replacer_id])
            for placeholder_id, replacer_id in self.placeholder_mapping.items()
        }


def test_option():
    text = "{|a|b|c}"
    answer = ["", "a", "b", "c"]
    p = WildcardProcessor(text)
    assert p.total() == len(answer)
    assert p.getn(0) == answer[0]
    assert p.getmany(4) == answer
    assert p.getmany(4, 1) == answer[1:]


def test_same():
    text = "{a|b},{a|b}"
    answer = ["a,a", "b,a", "a,b", "b,b"]
    p = WildcardProcessor(text)
    assert p.total() == len(answer)
    assert p.getn(0) == answer[0]
    assert p.getmany(4) == answer
    assert p.getmany(4, 1) == answer[1:]

