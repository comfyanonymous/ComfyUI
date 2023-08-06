
import re


class ParseError(Exception):
    def __init__(self, input, message):
        self.input = input.clone() # clone the parse cursor at the point of the error
        self.message = message
    
    def __str__(self):
        return f'{self.message} {self.input.loc()}'

class ParseLogicError(ParseError):
    # like a ParseError, in that it has an associated cursor position which will help in understanding the error
    # but unlike a ParseError, because it wasn't the user's fault
    # something that shouldn't be possible occurred in the code
    pass


class Cursor:
    def __init__(self, text, skip_space=False, consume=True, space=r'\s+'):
        self.text = text
        self.pos = 0        # current text position
        
        self.start = 0      # last match start position before whitespace skipping
        self.skip = 0       # last match start position after whitespace skipping
        self.end = 0        # last match end position
        
        self.skip_space = skip_space
        self.consume = consume
        self.space = space
    
    def prior(self):
        # returns a cursor pointing at the position prior to the last match
        prior = self.clone()
        prior.end = prior.start
        prior.pos = prior.start
        return prior
    
    def loc(self):
        # describe the cursor position in a human-readable form, suitable for error messages
        
        pos = self.pos
        text = self.text
        endline = re.compile(r'\n|$')
        
        # locate the line in which the current position is located
        line_start = 0
        line_id = 0
        while True:
            # determine line end position
            match = endline.search(text, pos=line_start)
            more_lines = match.group() == '\n'
            line_end = match.start()
            
            # we add 1 to include the newline in the positions covered (if present)
            # <<< at the end of the string, with no newline, it still kinda works okay I think
            if line_start <= pos < (line_end + 1):
                # pos is within the current line
                break
            
            if not more_lines:
                # pos is, somehow, somewhere past the end of the string
                # <<< for now, we'll just treat it as if pos was in the final line
                break
            
            line_start = line_end + 1 # skip newline
            line_id += 1
        
        line_size = line_end - line_start
        
        line_number = line_id + 1
        # line_offset is so ambiguous - is it offset *of* the line or offset of the cursor *within* the line?  in this case, it's the latter
        line_offset = pos - line_start
        line_text = text[line_start:line_end] # excludes newline
        caret_spacing = re.sub(r'[^\t]', ' ', line_text[:line_offset])
        
        return f'at line {line_number}, offset {line_offset}, line string {repr(line_text)}\n{line_text}\n{caret_spacing}^\n'
    
    def clone(self):
        # python's immutable strings should mean the actual string data for text is not copied
        clone = Cursor(self.text, skip_space=self.skip_space, consume=self.consume, space=self.space)
        # pos is the main purpose of the clone
        clone.pos = self.pos
        # this other stuff, we're just cloning for completeness
        clone.start = self.start
        clone.skip = self.skip
        clone.end = self.end
        return clone
    
    def string_match(self, string):
        '''
        Check for an exact match between the provided string and the input.
        Note that it's a string, not a regex.  Every character is literal.
        And it returns a bool, not a match object.
        '''
        pos = self.pos
        self.start = pos
        self.skip = pos
        self.end = pos
        size = len(string)
        if self.text[self.pos:self.pos + size] == string:
            pos += size
            self.pos = pos
            self.end = pos
            return True
        else:
            return False
    
    def match(self, regex, skip_space=None, consume=None, space=None):
        '''
        check if a regex matches at the cursor position
        given a match, update the cursor to consume the matched text (by default)
        Typical usage:
            if input.match(r'(\d+)'):
                # handle numbers
                value = int(input.m.group(1))
                # ...
            elif input.match(r'"'):
                # handle double-quoted strings
                # ...
            elif input.match(r'for'):
                # "for" loop
                # ...
            elif input.match(r'\s*$'):
                # end of input
                break
            else:
                raise 
        '''
        if skip_space == None:
            skip_space = self.skip_space
        if consume == None:
            consume = self.consume
        if space == None:
            space = self.space
        
        pos = self.pos
        self.start = pos
        self.skip = pos
        self.end = pos
        
        if skip_space:
            space_compile_flags = re.DOTALL
            space = re.compile(space, space_compile_flags) # <<< todo: compile once and reuse
            space_match = space.match(self.text, pos=pos)
            if space_match:
                pos = space_match.end()
                self.skip = pos
        
        compile_flags = re.DOTALL
        pattern = re.compile(regex, compile_flags)
        match = pattern.match(self.text, pos=pos)
        if match:
            pos = match.end()
            self.end = pos
            if consume:
                self.pos = pos
        
        return match
    
    def match_exact(self, regex, skip_space=False, consume=True):
        # check if a regex matches at the cursor position
        # consume the matched text (by default)
        # skip initial whitespace (by default)
        return self.match(regex, skip_space=skip_space, consume=consume)
    
    def check(self, regex, skip_space=None, consume=False):
        # check if a regex matches at the cursor position
        # do not consume the matched text (by default)
        # skip initial whitespace (by default)
        # another suitable name for this would have been "lookahead"
        return self.match(regex, skip_space=skip_space, consume=consume)

