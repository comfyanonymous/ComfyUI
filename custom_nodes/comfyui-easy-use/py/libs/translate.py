#credit to shadowcz007 for this module
#from https://github.com/shadowcz007/comfyui-mixlab-nodes/blob/main/nodes/TextGenerateNode.py
import re
import os
import folder_paths

import comfy.utils
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .utils import install_package
try:
    from lark import Lark, Transformer, v_args
except:
    print('install lark...')
    install_package('lark')
    from lark import Lark, Transformer, v_args

model_path = os.path.join(folder_paths.models_dir, 'prompt_generator')
zh_en_model_path = os.path.join(model_path, 'opus-mt-zh-en')
zh_en_model, zh_en_tokenizer = None, None

def correct_prompt_syntax(prompt=""):
    # print("input prompt",prompt)
    corrected_elements = []
    # 处理成统一的英文标点
    prompt = prompt.replace('（', '(').replace('）', ')').replace('，', ',').replace(';', ',').replace('。', '.').replace('：',':').replace('\\',',')
    # 删除多余的空格
    prompt = re.sub(r'\s+', ' ', prompt).strip()
    prompt = prompt.replace("< ","<").replace(" >",">").replace("( ","(").replace(" )",")").replace("[ ","[").replace(' ]',']')

    # 分词
    prompt_elements = prompt.split(',')

    def balance_brackets(element, open_bracket, close_bracket):
        open_brackets_count = element.count(open_bracket)
        close_brackets_count = element.count(close_bracket)
        return element + close_bracket * (open_brackets_count - close_brackets_count)

    for element in prompt_elements:
        element = element.strip()

        # 处理空元素
        if not element:
            continue

        # 检查并处理圆括号、方括号、尖括号
        if element[0] in '([':
            corrected_element = balance_brackets(element, '(', ')') if element[0] == '(' else balance_brackets(element, '[', ']')
        elif element[0] == '<':
            corrected_element = balance_brackets(element, '<', '>')
        else:
            # 删除开头的右括号或右方括号
            corrected_element = element.lstrip(')]')

        corrected_elements.append(corrected_element)

    # 重组修正后的prompt
    return  ','.join(corrected_elements)

def detect_language(input_str):
    # 统计中文和英文字符的数量
    count_cn = count_en = 0
    for char in input_str:
        if '\u4e00' <= char <= '\u9fff':
            count_cn += 1
        elif char.isalpha():
            count_en += 1

    # 根据统计的字符数量判断主要语言
    if count_cn > count_en:
        return "cn"
    elif count_en > count_cn:
        return "en"
    else:
        return "unknow"

def has_chinese(text):
    has_cn = False
    _text = text
    _text = re.sub(r'<.*?>', '', _text)
    _text = re.sub(r'__.*?__', '', _text)
    _text = re.sub(r'embedding:.*?$', '', _text)
    for char in _text:
        if '\u4e00' <= char <= '\u9fff':
            has_cn = True
            break
        elif char.isalpha():
            continue
    return has_cn

def translate(text):
    global zh_en_model_path, zh_en_model, zh_en_tokenizer

    if not os.path.exists(zh_en_model_path):
        zh_en_model_path = 'Helsinki-NLP/opus-mt-zh-en'

    if zh_en_model is None:

        zh_en_model = AutoModelForSeq2SeqLM.from_pretrained(zh_en_model_path).eval()
        zh_en_tokenizer = AutoTokenizer.from_pretrained(zh_en_model_path, padding=True, truncation=True)

    zh_en_model.to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        encoded = zh_en_tokenizer([text], return_tensors="pt")
        encoded.to(zh_en_model.device)
        sequences = zh_en_model.generate(**encoded)
        return zh_en_tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]

@v_args(inline=True)  # Decorator to flatten the tree directly into the function arguments
class ChinesePromptTranslate(Transformer):

    def sentence(self, *args):
        return ", ".join(args)

    def phrase(self, *args):
        return "".join(args)

    def emphasis(self, *args):
        # Reconstruct the emphasis with translated content
        return "(" + "".join(args) + ")"

    def weak_emphasis(self, *args):
        print('weak_emphasis:', args)
        return "[" + "".join(args) + "]"

    def embedding(self, *args):
        print('prompt embedding', args[0])
        if len(args) == 1:
            embedding_name = str(args[0])
            return f"embedding:{embedding_name}"
        elif len(args) > 1:
            embedding_name, *numbers = args

            if len(numbers) == 2:
                return f"embedding:{embedding_name}:{numbers[0]}:{numbers[1]}"
            elif len(numbers) == 1:
                return f"embedding:{embedding_name}:{numbers[0]}"
            else:
                return f"embedding:{embedding_name}"

    def lora(self, *args):
        if len(args) == 1:
            return f"<lora:{args[0]}>"
        elif len(args) > 1:
            # print('lora', args)
            _, loar_name, *numbers = args
            loar_name = str(loar_name).strip()
            if len(numbers) == 2:
                return f"<lora:{loar_name}:{numbers[0]}:{numbers[1]}>"
            elif len(numbers) == 1:
                return f"<lora:{loar_name}:{numbers[0]}>"
            else:
                return f"<lora:{loar_name}>"

    def weight(self, word, number):
        translated_word = translate(str(word)).rstrip('.')
        return f"({translated_word}:{str(number).strip()})"

    def schedule(self, *args):
        print('prompt schedule', args)
        data = [str(arg).strip() for arg in args]

        return f"[{':'.join(data)}]"

    def word(self, word):
        # Translate each word using the dictionary
        word = str(word)
        match_cn = re.search(r'@.*?@', word)
        if re.search(r'__.*?__', word):
            return word.rstrip('.')
        elif match_cn:
            chinese = match_cn.group()
            before = word.split('@', 1)
            before = before[0] if len(before) > 0 else ''
            before = translate(str(before)).rstrip('.') if before else ''
            after = word.rsplit('@', 1)
            after = after[len(after)-1] if len(after) > 1 else ''
            after = translate(after).rstrip('.') if after else ''
            return before + chinese.replace('@', '').rstrip('.') + after
        elif detect_language(word) == "cn":
            return translate(word).rstrip('.')
        else:
            return word.rstrip('.')


#定义Prompt文法
grammar = r"""
start: sentence
sentence: phrase ("," phrase)*
phrase: emphasis | weight | word | lora | embedding | schedule 
emphasis: "(" sentence ")" -> emphasis
        | "[" sentence "]" -> weak_emphasis
weight: "(" word ":" NUMBER ")"
schedule: "[" word ":" word ":" NUMBER "]"
lora: "<" WORD ":" WORD (":" NUMBER)? (":" NUMBER)? ">"
embedding: "embedding" ":" WORD (":" NUMBER)? (":" NUMBER)?
word: WORD

NUMBER: /\s*-?\d+(\.\d+)?\s*/
WORD: /[^,:\(\)\[\]<>]+/
"""
def zh_to_en(text):
    global zh_en_model_path, zh_en_model, zh_en_tokenizer
    # 进度条
    pbar = comfy.utils.ProgressBar(len(text) + 1)
    texts = [correct_prompt_syntax(t) for t in text]

    install_package('sentencepiece', '0.2.0')

    if not os.path.exists(zh_en_model_path):
        zh_en_model_path = 'Helsinki-NLP/opus-mt-zh-en'

    if zh_en_model is None:
        zh_en_model = AutoModelForSeq2SeqLM.from_pretrained(zh_en_model_path).eval()
        zh_en_tokenizer = AutoTokenizer.from_pretrained(zh_en_model_path, padding=True, truncation=True)

    zh_en_model.to("cuda" if torch.cuda.is_available() else "cpu")

    prompt_result = []

    en_texts = []

    for t in texts:
        if t:
            # translated_text =  translated_word = translate(zh_en_tokenizer,zh_en_model,str(t))
            parser = Lark(grammar, start="start", parser="lalr", transformer=ChinesePromptTranslate())
            # print('t',t)
            result = parser.parse(t).children
            # print('en_result',result)
            # en_text=translate(zh_en_tokenizer,zh_en_model,text_without_syntax)
            en_texts.append(result[0])

    zh_en_model.to('cpu')
    # print("test en_text", en_texts)
    # en_text.to("cuda" if torch.cuda.is_available() else "cpu")

    pbar.update(1)
    for t in en_texts:
        prompt_result.append(t)
        pbar.update(1)

    # print('prompt_result', prompt_result, )
    if len(prompt_result) == 0:
        prompt_result = [""]

    return prompt_result