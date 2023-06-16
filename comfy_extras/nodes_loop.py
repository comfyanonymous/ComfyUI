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

        return (kwargs['loop_condition'].get_next(kwargs['initial_input'], current), )


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


NODE_CLASS_MAPPINGS = {
    "LoopControl": LoopControl,
    "LoopCounterCondition": LoopCounterCondition,
}
