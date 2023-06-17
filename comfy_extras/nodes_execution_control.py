class LoopControl:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "loop_condition": ("LOOP_CONDITION", ),
                        "initial_input": ("*", ),
                        "loopback_input": ("*", ),
                    },
                }

    RETURN_TYPES = ("*", )
    FUNCTION = "doit"

    def doit(s, **kwargs):
        if 'loopback_input' not in kwargs or kwargs['loopback_input'] is None:
            current = kwargs['initial_input']
        else:
            current = kwargs['loopback_input']

        result = kwargs['loop_condition'].get_next(kwargs['initial_input'], current)
        if result is None:
            return None
        else:
            return (result, )


class CounterCondition:
    def __init__(self, value):
        self.max = value
        self.current = 0

    def get_next(self, initial_value, value):
        print(f"CounterCondition: {self.current}/{self.max}")

        self.current += 1
        if self.current == 1:
            return initial_value
        elif self.current <= self.max:
            return value
        else:
            return None


class LoopCounterCondition:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "count": ("INT", {"default": 1, "min": 0, "max": 9999999, "step": 1}),
                        "trigger": (["A", "B"], )
                    },
                }

    RETURN_TYPES = ("LOOP_CONDITION", )
    FUNCTION = "doit"

    def doit(s, count, trigger):
        return (CounterCondition(count), )


# To facilitate the use of multiple inputs as loopback inputs, InputZip and InputUnzip are provided.
class InputZip:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "input1": ("*", ),
                        "input2": ("*", ),
                    },
                }

    RETURN_TYPES = ("*", )
    FUNCTION = "doit"

    def doit(s, input1, input2):
        return ((input1, input2), )


class InputUnzip:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "zipped_input": ("*", ),
                    },
                }

    RETURN_TYPES = ("*", "*", )
    FUNCTION = "doit"

    def doit(s, zipped_input):
        input1, input2 = zipped_input
        return (input1, input2, )


class ExecutionBlocker:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "input": ("*", ),
                        "signal": ("*", ),
                     },
                }

    RETURN_TYPES = ("*", )
    FUNCTION = "doit"

    def doit(s, input, signal):
        return input


class ExecutionOneOf:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                        "input1": ("*", ),
            },
            "optional": {
                        "input2": ("*", ),
                        "input3": ("*", ),
                        "input4": ("*", ),
                        "input5": ("*", ),
                     },
               }

    RETURN_TYPES = ("*", )
    FUNCTION = "doit"

    def doit(s, **kwargs):
        if 'input1' in kwargs and kwargs['input1'] is not None:
            return (kwargs['input1'], )
        elif 'input2' in kwargs and kwargs['input2'] is not None:
            return (kwargs['input2'], )
        elif 'input3' in kwargs and kwargs['input3'] is not None:
            return (kwargs['input3'],)
        elif 'input4' in kwargs and kwargs['input4'] is not None:
            return (kwargs['input4'],)
        elif 'input5' in kwargs and kwargs['input5'] is not None:
            return (kwargs['input5'],)
        else:
            return None


class ExecutionSwitch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "select": ("INT", {"default": 1, "min": 0, "max": 5}),
                        "input1": ("*", ),
                    },
                "optional": {
                        "input2_opt": ("*", ),
                        "input3_opt": ("*", ),
                        "input4_opt": ("*", ),
                        "input5_opt": ("*", ),
                    }
                }

    RETURN_TYPES = ("*", "*", "*", "*", "*", )
    FUNCTION = "doit"

    def doit(s, select, input1, input2_opt=None, input3_opt=None, input4_opt=None, input5_opt=None):
        if select == 1:
            return input1, None, None, None, None
        elif select == 2:
            return None, input2_opt, None, None, None
        elif select == 3:
            return None, None, input3_opt, None, None
        elif select == 4:
            return None, None, None, input4_opt, None
        elif select == 5:
            return None, None, None, None, input5_opt
        else:
            return None, None, None, None, None


NODE_CLASS_MAPPINGS = {
    "ExecutionSwitch": ExecutionSwitch,
    "ExecutionBlocker": ExecutionBlocker,
    "ExecutionOneOf": ExecutionOneOf,
    "LoopControl": LoopControl,
    "LoopCounterCondition": LoopCounterCondition,
    "InputZip": InputZip,
    "InputUnzip": InputUnzip,
}
