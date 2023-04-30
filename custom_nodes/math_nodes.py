class FrameCounter:
    def __init__(self, event_dispatcher):
        self.event_dispatcher = event_dispatcher

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frame": ("INT", {"default": 0}),
                "fired": ("BOOL", {"default": False}),
            },
        }
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return True

    RETURN_TYPES = ("text",)
    FUNCTION = "frame_counter"

    CATEGORY = "math"

    def frame_counter(self, frame, fired):
        if fired:
            frame += 1
        return (frame,)

NODE_CLASS_MAPPINGS = {
    "FrameCounter": FrameCounter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FrameCounter": "Frame Counter",
}
