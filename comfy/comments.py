
import re

def strip_c_comments(text, strict=True):
    # Processes the text and strips out any C-style block "/* ... */" or line "// ..." comments found.
    # from old dynamicPrompts.js: return str.replace(/\/\*[\s\S]*?\*\/|\/\/.*/g,'');
    return re.sub(r'/\*[.\n]*?(?:\*/|$)|//.*', '', text)

def strip_hash_comments(text, strict=True):
    # Processes the text and strips out any hash "# ... " comments found.
    return re.sub(r'#.*', '', text)

