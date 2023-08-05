
import comfy.parse as parse
from comfy.parse import ParseError, ParseLogicError

def strip_c_comments(text, strict=True):
	'''
	Processes the text and strips out any C-style block "/* ... */" or line "// ..." comments found.
	'''
	out = []
	input = parse.Cursor(text)
	while True:
		if 0: pass
		elif input.match(r'\/'):
			# C-style block "/* */" or line "//" comment
			comment = input.prior()
			if 0: pass
			elif m := input.match(r'\/'):
				# // line comment
				if not input.match(r'.*?(?:\n|$)'):
					if strict:
						raise ParseLogicError(comment, f"Failed to find end of C-style // line comment")
					input.match(r'.*') # consume unterminated comment (however that might be possible)
				out.append('\n')
			elif m := input.match(r'\*'):
				# /* ... */ block comment
				if not input.match(r'.*?\*\/'):
					if strict:
						raise ParseError(comment, f"Unterminated C-style /* ... */ block comment")
					input.match(r'.*') # consume unterminated comment
				out.append(' ')
			else:
				# it was a literal /, not a comment after all
				out.append('/');
		elif m := input.match(r'[^/]+'):
			out.append(m.group(0))
		elif input.match(r'$'):
			break;
		else:
			raise ParseLogicError(input, f"Failed to match")
	
	return ''.join(out)

def strip_hash_comments(text, strict=True):
	'''
	Processes the text and strips out any hash "# ... " comments found.
	'''
	out = []
	input = parse.Cursor(text)
	while True:
		if 0: pass
		elif input.match(r'\#'):
			comment = input.prior()
			# scripting-language-style "# ...\n" comment
			if not input.match(r'.*?(?:\n|$)'):
				if strict:
					raise ParseLogicError(comment, f"Failed to find end of script-style # ... comment")
				input.match(r'.*') # consume unterminated comment (however that might be possible)
			out.append('\n')
		elif m := input.match(r'[^#]+'):
			out.append(m.group(0))
		elif input.match(r'$'):
			break;
		else:
			raise ParseLogicError(input, f"Failed to match")
	
	return ''.join(out)

