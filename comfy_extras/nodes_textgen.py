from comfy_api.latest import ComfyExtension, io
from typing_extensions import override
import comfy.utils
import math

class TextGenerateGemma3(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        # Define dynamic combo options for sampling mode
        sampling_options = [
            io.DynamicCombo.Option(
                key="off",
                inputs=[]
            ),
            io.DynamicCombo.Option(
                key="on",
                inputs=[
                    io.Float.Input("temperature", default=1.0, min=0.01, max=2.0, step=0.000001),
                    io.Int.Input("top_k", default=64, min=0, max=1000),
                    io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01),
                    io.Float.Input("min_p", default=0.05, min=0.0, max=1.0, step=0.01),
                    io.Float.Input("repetition_penalty", default=1.05, min=0.0, max=5.0, step=0.01),
                    io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                ]
            ),
        ]

        return io.Schema(
            node_id="TextGenerateGemma3",
            category="textgen/",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("system_prompt", multiline=True, force_input=True, optional=True, default="You are a helpful assistant."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Image.Input("image", optional=True),
                io.Int.Input("max_length", default=256, min=1, max=2048),
                io.DynamicCombo.Input("sampling_mode", options=sampling_options, display_name="Sampling Mode"),
            ],
            outputs=[
                io.String.Output(display_name="generated_text"),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, max_length, sampling_mode, image=None, system_prompt="You are a helpful assistant.") -> io.NodeOutput:

        if image is None:
            images = []
        else:
            samples = image.movedim(-1, 1)
            total = int(896 * 896)

            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)

            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            image = s.movedim(1, -1)
            images = [image[:, :, :, :3]]

        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        if system_prompt and system_prompt.strip():
            if images:
                formatted_prompt = f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n<start_of_turn>user\n<image_soft_token>{prompt}<end_of_turn>\n<start_of_turn>model\n"
            else:
                formatted_prompt = f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

        tokens = clip.tokenize(formatted_prompt, images=images)

        # Get sampling parameters from dynamic combo
        do_sample = sampling_mode.get("sampling_mode") == "on"
        temperature = sampling_mode.get("temperature", 1.0)
        top_k = sampling_mode.get("top_k", 50)
        top_p = sampling_mode.get("top_p", 1.0)
        min_p = sampling_mode.get("min_p", 0.0)
        seed = sampling_mode.get("seed", None)
        repetition_penalty = sampling_mode.get("repetition_penalty", 1.0)

        generated_ids = clip.generate(
            tokens,
            do_sample=do_sample,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            seed=seed
        )

        generated_text = clip.decode(generated_ids[0], skip_special_tokens=True)
        return io.NodeOutput(generated_text)


class TextgenExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TextGenerateGemma3,
        ]

async def comfy_entrypoint() -> TextgenExtension:  # ComfyUI calls this to load your extension and its nodes.
    return TextgenExtension()
