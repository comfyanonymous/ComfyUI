class Options:
    img2img_background_color = "#ffffff"  # Set to white for now


class State:
    interrupted = False

    def begin(self):
        pass

    def end(self):
        pass


opts = Options()
state = State()

# Will only ever hold 1 upscaler
sd_upscalers = [None]
# The upscaler usable by ComfyUI nodes
actual_upscaler = None

# Batch of images to upscale
batch = None
batch_as_tensor = None
