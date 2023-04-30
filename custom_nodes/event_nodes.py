class EventListener:
    def __init__(self, event_dispatcher):
        self.event_dispatcher = event_dispatcher

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "event_type": (["node_started", "node_finished"],),
                "class_type": ("STRING", {"default": "KSampler"})
            },
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return True

    RETURN_TYPES = ("BOOL",)

    FUNCTION = "listen"

    CATEGORY = "Events"

    def listen(self, event_type, class_type):
        self._fired = False

        def event_listener(event, event_data):
            print(f"Got an event of type {event_data['event_type']} with data {event_data}")
            if (event_data["event_type"] == event_type and event_data["class_type"] == class_type):
                self._fired = True

        self.event_dispatcher.subscribe(event_type, event_listener)

        return (self._fired,)


NODE_CLASS_MAPPINGS = {
    "EventListener": EventListener,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EventListener": "Event Listener",
}