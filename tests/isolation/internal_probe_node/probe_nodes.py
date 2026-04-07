from __future__ import annotations


class InternalIsolationProbeImage:
    CATEGORY = "tests/isolation"
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def run(self):
        from comfy_api.latest import UI
        import torch

        image = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
        image[:, :, :, 0] = 1.0
        ui = UI.PreviewImage(image)
        return {"ui": ui.as_dict(), "result": ()}


class InternalIsolationProbeAudio:
    CATEGORY = "tests/isolation"
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def run(self):
        from comfy_api.latest import UI
        import torch

        waveform = torch.zeros((1, 1, 32), dtype=torch.float32)
        audio = {"waveform": waveform, "sample_rate": 44100}
        ui = UI.PreviewAudio(audio)
        return {"ui": ui.as_dict(), "result": ()}


class InternalIsolationProbeUI3D:
    CATEGORY = "tests/isolation"
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def run(self):
        from comfy_api.latest import UI
        import torch

        bg_image = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
        bg_image[:, :, :, 1] = 1.0
        camera_info = {"distance": 1.0}
        ui = UI.PreviewUI3D("internal_probe_preview.obj", camera_info, bg_image=bg_image)
        return {"ui": ui.as_dict(), "result": ()}


NODE_CLASS_MAPPINGS = {
    "InternalIsolationProbeImage": InternalIsolationProbeImage,
    "InternalIsolationProbeAudio": InternalIsolationProbeAudio,
    "InternalIsolationProbeUI3D": InternalIsolationProbeUI3D,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InternalIsolationProbeImage": "Internal Isolation Probe Image",
    "InternalIsolationProbeAudio": "Internal Isolation Probe Audio",
    "InternalIsolationProbeUI3D": "Internal Isolation Probe UI3D",
}
