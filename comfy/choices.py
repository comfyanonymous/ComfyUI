

import os
import random

import comfy.parse as parse
from comfy.parse import ParseError, ParseLogicError

from comfy.comments import strip_c_comments

def translate_choices_with_c_comments(text, seed=None, strict=True, reescape=frozenset()):
    text = strip_c_comments(text, strict=strict)
    text = translate(text, seed=seed, strict=strict, reescape = reescape)
    return text

def get_random_seed():
    return int.from_bytes(os.urandom(8))

def translate(text, seed=None, strict=True, reescape=frozenset()):
    '''
    Parses the text, translating "{A|B|C}" choices into a single chosen option.
    An option is chosen randomly from the available options.
    For example: "a {green|red|blue} ball on a {wooden|metal} bench" might expand to "a red ball on a wooden bench".
    Nesting choices is supported, so 
    "a woman wearing a {{lavish|garish|expensive|stylish|} {red|brown|blue|} dress|{sexy|realistic|} {police|nurse|maid} uniform|{black leather|wooly|thick} coat}"
    could expand to
    "a woman wearing a realistic police uniform".
    All random choices are governed by the supplied random seed value, ensuring repeatability.
    
    If strict is True, exceptions will be thrown if the input doesn't conform to expectations.
    
    reescape indicates the set of metacharacters that, if escaped with a backslash in the input, should be re-escaped in the output.
    This is useful to avoid need for multi-escaping when incorporating this parser as a single phase in a multi-phase parsing operation.
    Note that while the default is a frozenset, you can pass anything that works with the "in" operator, such as a string or a set.
    '''
    
    def parse_choice(input):
        options = []
        while True:
            options.append(parse_text_with_choices(input))
            if m := input.match(r'\|'):
                # loop around for another choice
                pass
            else:
                # at this point, the input must be }
                # although for incorrectly-formed input, it could be end of input too
                # regardless, the correct action here is to break and return to the caller
                break
        
        # choose one of the options
        text = rng.choice(options)
        return text
    
    def parse_text_with_choices(input):
        out = []
        
        while True:
            if 0: pass
            elif m := input.match(r'\\'):
                # \ = escape character
                if m := input.match(r'.'):
                    ch = m.group(0)
                    if ch in reescape:
                        out.append('\\')
                    out.append(ch)
                else:
                    if strict:
                        raise ParseError(input, f'Unexpected end of input after backslash')
            elif m := input.match(r'\{'):
                # { ... | ... } choice
                openbrace = input.prior()
                chosen_text = parse_choice(input)
                if not input.match(r'\}'):
                    if strict:
                        raise ParseError(openbrace, f"Missing matching closing brace '}}' for earlier open brace '{{'")
                out.append(chosen_text)
            elif m := input.match(r'[^\\\{\}\|]+'):
                # 1 or more non-metacharacters
                out.append(m.group(0))
            else:
                # didn't match \, {, / or non-metacharacters
                # must be either |, } or end of input
                break
        
        return ''.join(out)
    
    def parse_text_with_choices_outer(input):
        # this function and the contained loop is required to support the non-strict parsing mode
        # it catches the case where we exit parse_text_with_choices upon encountering | or }, and don't find ourselves withing a calling instance of parse_choice
        out = []
        while True:
            out.append(parse_text_with_choices(input))
            if 0:pass
            elif input.match(r'$'):
                break
            elif input.match(r'\|'):
                if strict:
                    raise ParseError(input.prior(), f"Encountered a choice delimiter '|' outside any choice block")
            elif input.match(r'\}'):
                if strict:
                    raise ParseError(input.prior(), f"Encountered a closing brace '}}' without a matching open brace")
            else:
                if strict:
                    raise ParseLogicError(input, f'Failed to parse up to the end of the prompt text')
                break
        
        return ''.join(out)
    
    
    if seed == None:
        seed = get_random_seed()
    
    # init our local random number generator
    rng = random.Random(seed)
    
    input = parse.Cursor(text)
    out = parse_text_with_choices_outer(input)
    return out

