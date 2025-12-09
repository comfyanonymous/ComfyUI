import pytest
from comfy_execution.graph_utils import GraphBuilder
from comfy.client.embedded_comfy_client import Comfy
from comfy.api.components.schema.prompt import Prompt

class TestQwenVLVideo:
    @pytest.mark.asyncio
    async def test_qwenvl_video_loading(self):
        graph = GraphBuilder()

        # Load QwenVL model (using a small one as requested)
        # Qwen/Qwen2-VL-2B-Instruct is a good candidate for a "small" QwenVL model
        model_loader = graph.node("TransformersLoader1", ckpt_name="Qwen/Qwen2-VL-2B-Instruct")

        # Load video from URL with frame cap to avoid OOM
        video_url = "https://upload.wikimedia.org/wikipedia/commons/f/f7/2024-04-05_Luisenpark_MA_Ziegen_2.webm"
        load_video = graph.node("LoadVideoFromURL", value=video_url, frame_load_cap=16, select_every_nth=10)

        # Tokenize with video
        # OneShotInstructTokenize has optional 'videos' input
        tokenizer = graph.node("OneShotInstructTokenize", model=model_loader.out(0), prompt="Describe this video.", videos=load_video.out(0), chat_template="default")

        # Generate
        generation = graph.node("TransformersGenerate", model=model_loader.out(0), tokens=tokenizer.out(0), max_new_tokens=50, seed=42)

        # OmitThink (as requested)
        omit_think = graph.node("OmitThink", value=generation.out(0))

        # Save output
        graph.node("SaveString", value=omit_think.out(0), filename_prefix="qwenvl_video_test")

        workflow = graph.finalize()
        prompt = Prompt.validate(workflow)

        async with Comfy() as client:
            outputs = await client.queue_prompt(prompt)
            
        # We expect it to fail before this, but if it succeeds, we should check the output
        assert len(outputs) > 0
