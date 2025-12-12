import pytest
from comfy_execution.graph_utils import GraphBuilder
from comfy.client.embedded_comfy_client import Comfy
from comfy.api.components.schema.prompt import Prompt

class TestMixedMediaGeneric:
    @pytest.mark.asyncio
    async def test_mixed_media_generic(self):
        graph = GraphBuilder()

        # Load BLIP (small, standard model, image-only processor)
        model_loader = graph.node("TransformersLoader1", ckpt_name="Salesforce/blip-image-captioning-base")

        # Load video (Goat)
        video_url = "https://upload.wikimedia.org/wikipedia/commons/f/f7/2024-04-05_Luisenpark_MA_Ziegen_2.webm"
        # Use frame cap to keep it light
        load_video = graph.node("LoadVideoFromURL", value=video_url, frame_load_cap=16, select_every_nth=10)

        # Load image (Worm)
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Earthworm.jpg/330px-Earthworm.jpg"
        load_image = graph.node("LoadImageFromURL", value=image_url)

        # Tokenize with both video and image
        # BLIP expects "images" (list of tensors) if we use the processor correctly.
        # My fallback logic should convert video frames to images.
        tokenizer = graph.node("OneShotInstructTokenize", model=model_loader.out(0), prompt="a photography of", videos=load_video.out(0), images=load_image.out(0), chat_template="default")

        # Generate
        generation = graph.node("TransformersGenerate", model=model_loader.out(0), tokens=tokenizer.out(0), max_new_tokens=100, seed=42)

        # OmitThink
        omit_think = graph.node("OmitThink", value=generation.out(0))

        # Save output
        graph.node("SaveString", value=omit_think.out(0), filename_prefix="mixed_media_test")

        workflow = graph.finalize()
        prompt = Prompt.validate(workflow)

        from comfy.cli_args import default_configuration
        config = default_configuration()
        config.enable_video_to_image_fallback = True

        async with Comfy(configuration=config) as client:
            outputs = await client.queue_prompt(prompt)
            
        assert len(outputs) > 0
