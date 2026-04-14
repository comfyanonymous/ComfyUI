from __future__ import annotations

import torch
from typing_extensions import override

from comfy_api.latest import ComfyExtension, Input, io, ui
from comfy_execution.progress import get_progress_state
from comfy_execution.utils import get_executing_context
from server import PromptServer


class FrameProgressTracker:
    def __init__(self):
        self._last_reported=None

    def emit(self, frames_processed):
        if frames_processed == self._last_reported:
            return

        current = get_executing_context()
        server = getattr(PromptServer, "instance", None)
        if current is None or server is None or server.client_id is None:
            return

        server.send_progress_text(
            f"processed {frames_processed} frames",
            current.node_id,
            server.client_id,
        )
        self._last_reported = frames_processed


def drain_image_stream(stream, chunk_size, progress):
    stream.reset()
    frames_processed = 0

    while True:
        progress.emit(frames_processed)
        chunk = stream.pull(chunk_size)
        frames_processed += int(chunk.shape[0])
        if chunk.shape[0] < chunk_size:
            progress.emit(frames_processed)
            return frames_processed

class TensorImageStream(Input.ImageStream):
    """Simple IMAGE_STREAM backed by a materialized IMAGE batch tensor."""

    def __init__(self, images: Input.Image):
        super().__init__()
        self._images = images
        self._index = 0
        self._total_frames = int(images.shape[0])
        self._progress_started = False

    def _update_progress(self, value: int) -> None:
        current = get_executing_context()
        if current is None:
            return
        get_progress_state().update_progress(
            current.node_id,
            value=float(value),
            max_value=float(max(self._total_frames, 1)),
        )

    def get_dimensions(self) -> tuple[int, int]:
        return self._images.shape[2], self._images.shape[1]

    def do_reset(self) -> None:
        self._index = 0
        self._progress_started = False

    def do_pull(self, max_frames: int) -> Input.Image:
        if not self._progress_started:
            self._update_progress(0)
            self._progress_started = True

        start = self._index
        end = min(start + max_frames, self._images.shape[0])
        self._index = end
        chunk = self._images[start:end].clone()
        self._update_progress(end)
        return chunk


class PreviewingImageStream(Input.ImageStream):
    def __init__(self, stream: Input.ImageStream):
        super().__init__()
        self._stream = stream

    def _emit_preview(self, chunk: Input.Image) -> None:
        if int(chunk.shape[0]) == 0:
            return

        current = get_executing_context()
        if current is None:
            return

        server = getattr(PromptServer, "instance", None)
        if server is None or server.client_id is None:
            return

        preview_output = ui.PreviewImage(chunk[-1:]).as_dict()
        server.send_sync(
            "executed",
            {
                "node": current.node_id,
                "display_node": current.node_id,
                "output": preview_output,
                "prompt_id": current.prompt_id,
            },
            server.client_id,
        )

    def get_dimensions(self) -> tuple[int, int]:
        return self._stream.get_dimensions()

    def do_reset(self) -> None:
        self._stream.reset()

    def do_pull(self, max_frames: int) -> Input.Image:
        chunk = self._stream.pull(max_frames)
        self._emit_preview(chunk)
        return chunk


class VAEDecodedImageStream(Input.ImageStream):
    def __init__(self, vae, latent: Input.Latent):
        super().__init__()
        self._vae = vae
        self._latent = latent
        vae.throw_exception_if_invalid()
        if not getattr(vae.first_stage_model, "comfy_has_chunked_io", False):
            raise RuntimeError("This VAE does not expose chunked decode support, so VAE Decode Stream cannot be used.")
        if latent.ndim != 5:
            raise RuntimeError("VAE Decode Stream expects a video latent shaped [batch, channels, frames, height, width].")
        if latent.shape[0] != 1:
            raise RuntimeError("VAE Decode Stream currently requires latent batch size 1.")
        output_shape = vae.decode_output_shape(latent.shape)
        self._channels = int(output_shape[1])
        self._width = int(output_shape[4])
        self._height = int(output_shape[3])
        self._total_frames = int(output_shape[0] * output_shape[2])
        self._frames_emitted = 0

    def _update_progress(self, value: int) -> None:
        current = get_executing_context()
        if current is None:
            return
        get_progress_state().update_progress(
            current.node_id,
            value=float(value),
            max_value=float(max(self._total_frames, 1)),
        )

    def get_dimensions(self) -> tuple[int, int]:
        return self._width, self._height

    def do_reset(self) -> None:
        self._frames_emitted = 0
        self._update_progress(0)
        self._vae.decode_stream_start(self._latent)

    def do_pull(self, max_frames: int) -> Input.Image:
        chunk = self._vae.first_stage_model.decode_chunk(max_frames)
        if chunk is None:
            return torch.empty(
                (0, self._height, self._width, self._channels),
                device=self._vae.output_device,
                dtype=self._vae.vae_output_dtype(),
            )

        chunk = chunk.to(device=self._vae.output_device, dtype=self._vae.vae_output_dtype())
        chunk = self._vae.process_output(chunk).movedim(1, -1)
        chunk = chunk.reshape((-1,) + tuple(chunk.shape[-3:]))
        self._frames_emitted += int(chunk.shape[0])
        self._update_progress(self._frames_emitted)
        return chunk


class ImageBatchToStream(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageBatchToStream",
            display_name="Image Batch To Stream",
            category="image/stream",
            search_aliases=["image to stream", "batch to stream", "frames to stream"],
            description="Wraps a batched IMAGE tensor as a pull-based IMAGE_STREAM.",
            inputs=[
                io.Image.Input("image", tooltip="A batched IMAGE tensor in BHWC format."),
            ],
            outputs=[
                io.ImageStream.Output(display_name="stream"),
            ],
        )

    @classmethod
    def execute(cls, image: Input.Image) -> io.NodeOutput:
        return io.NodeOutput(TensorImageStream(image))


class ImageStreamToBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageStreamToBatch",
            display_name="Image Stream To Batch",
            category="image/stream",
            search_aliases=["stream to image", "stream to batch", "collect stream"],
            description="Materializes an IMAGE_STREAM back into a batched IMAGE tensor.",
            inputs=[
                io.ImageStream.Input("stream", tooltip="A pull-based IMAGE_STREAM."),
                io.Int.Input("batch_size", default=4096, min=1, max=4096),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, stream: Input.ImageStream, batch_size: int) -> io.NodeOutput:
        chunks: list[Input.Image] = []
        stream.reset()

        while True:
            chunk = stream.pull(batch_size)

            chunks.append(chunk)
            if chunk.shape[0] < batch_size:
                break

        return io.NodeOutput(torch.cat(chunks, dim=0))


class VAEDecodeStream(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VAEDecodeStream",
            display_name="VAE Decode Stream",
            category="image/stream",
            search_aliases=["vae stream decode", "latent to stream", "video latent stream"],
            description="Decodes a latent into an IMAGE_STREAM.",
            inputs=[
                io.Latent.Input("samples", tooltip="The LTX latent to decode."),
                io.Vae.Input("vae", tooltip="The LTX VAE used for chunked streaming decode."),
            ],
            outputs=[
                io.ImageStream.Output(display_name="stream"),
            ],
        )

    @classmethod
    def execute(cls, samples: Input.Latent, vae) -> io.NodeOutput:
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]
        return io.NodeOutput(VAEDecodedImageStream(vae, latent))


class PreviewImageStream(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PreviewImageStream",
            display_name="Preview Image Stream",
            category="image/stream",
            search_aliases=["stream preview", "preview frames", "preview image stream"],
            description="Passes an IMAGE_STREAM through while previewing the last frame from each pulled chunk.",
            has_intermediate_output=True,
            inputs=[
                io.ImageStream.Input("stream", tooltip="The image stream to preview inline."),
            ],
            outputs=[
                io.ImageStream.Output(display_name="passthrough"),
            ],
        )

    @classmethod
    def execute(cls, stream: Input.ImageStream) -> io.NodeOutput:
        return io.NodeOutput(PreviewingImageStream(stream))


class StreamSink(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StreamSink",
            search_aliases=["consume stream", "drain stream", "image stream sink"],
            display_name="Stream Sink",
            category="image/stream",
            description="Consumes an IMAGE_STREAM by pulling it to EOF.",
            inputs=[
                io.ImageStream.Input("stream", tooltip="The image stream to consume."),
                io.Int.Input("chunk_size", default=8, min=1, max=4096),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, stream: Input.ImageStream, chunk_size: int) -> io.NodeOutput:
        drain_image_stream(stream, chunk_size, progress=FrameProgressTracker())
        return io.NodeOutput()


class ImageStreamExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ImageBatchToStream,
            ImageStreamToBatch,
            VAEDecodeStream,
            PreviewImageStream,
            StreamSink,
        ]


async def comfy_entrypoint() -> ImageStreamExtension:
    return ImageStreamExtension()
