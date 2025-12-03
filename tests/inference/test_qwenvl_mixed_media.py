import pytest
from comfy_execution.graph_utils import GraphBuilder
from comfy.client.embedded_comfy_client import Comfy
from comfy.api.components.schema.prompt import Prompt

class TestQwenVLMixedMedia:
    @pytest.mark.asyncio
    async def test_qwenvl_mixed_media(self):
        graph = GraphBuilder()

        # Load Qwen2-VL-2B-Instruct
        model_loader = graph.node("TransformersLoader1", ckpt_name="Qwen/Qwen2-VL-2B-Instruct")

        # Load video (Goat)
        video_url = "https://upload.wikimedia.org/wikipedia/commons/f/f7/2024-04-05_Luisenpark_MA_Ziegen_2.webm"
        # Use frame cap to keep it light
        load_video = graph.node("LoadVideoFromURL", value=video_url, frame_load_cap=16, select_every_nth=10)

        # Load image (Worm)
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Earthworm.jpg/330px-Earthworm.jpg"
        load_image = graph.node("LoadImageFromURL", value=image_url)

        # Tokenize with both video and image
        tokenizer = graph.node("OneShotInstructTokenize", model=model_loader.out(0), prompt="Describe what you see in the video and the image.", videos=load_video.out(0), images=load_image.out(0), chat_template="default")

        # Generate
        generation = graph.node("TransformersGenerate", model=model_loader.out(0), tokens=tokenizer.out(0), max_new_tokens=100, seed=42)

        # OmitThink
        omit_think = graph.node("OmitThink", value=generation.out(0))

        # Save output
        graph.node("SaveString", value=omit_think.out(0), filename_prefix="qwenvl_mixed_media_test")

        workflow = graph.finalize()
        prompt = Prompt.validate(workflow)

        async with Comfy() as client:
            outputs = await client.queue_prompt(prompt)
            
        assert len(outputs) > 0
