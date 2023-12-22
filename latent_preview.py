import torch
import queue
from PIL import Image
import struct
import numpy as np
from comfy.cli_args import args, LatentPreviewMethod
from comfy.taesd.taesd import TAESD
import folder_paths
import comfy.utils

MAX_PREVIEW_RESOLUTION = 512

class LatentPreviewer:
    def decode_latent_to_preview(self, x0):
        pass

    def decode_latent_to_preview_image(self, preview_format, x0):
        preview_image = self.decode_latent_to_preview(x0)
        return ("JPEG", preview_image, MAX_PREVIEW_RESOLUTION)

class LazyTAESDPreviewerImpl(LatentPreviewer):
    blank_preview = Image.new(mode="RGB", size = (MAX_PREVIEW_RESOLUTION, MAX_PREVIEW_RESOLUTION), color = (20, 40, 40))

    @classmethod
    def worker_fun(cls, taesd, pending, ready):
        while True:
            wi = None
            (wi, _) = cls.snarf_all(pending, blocking = True)
            if wi is None:
                break
            with torch.no_grad():
                x_sample = taesd.decode(wi)[0].detach()
                x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)
            try:
                ready.put_nowait(Image.fromarray(x_sample))
            except queue.Full:
                print("warning: Lazy TAESD: Worker cannot submit - queue full")

    def __init__(self, taesd):
        import threading
        self.device = torch.device("cpu")
        self.pending = queue.Queue(4)
        self.ready = queue.Queue(4)
        taesd = taesd.to(self.device)
        taesd.share_memory()
        self.worker = threading.Thread(target = self.worker_fun, args = (taesd, self.pending, self.ready))
        self.worker.start()
        self.last_preview = self.blank_preview

    def __del__(self):
        self.snarf_all(self.pending)
        self.snarf_all(self.ready)
        self.pending.put(None, True)
        if self.worker.is_alive():
            self.worker.join()

    @staticmethod
    def snarf_all(chan, blocking = False):
        item = None
        have_item = False
        try:
            while True:
                item = chan.get_nowait()
                have_item = True
                chan.task_done()
        except queue.Empty:
            pass
        if blocking and not have_item:
            return (chan.get(True), True)
        return (item, have_item)

    def decode_latent_to_preview_image(self, preview_format, x0, blocking = False):
        preview_image = self.decode_latent_to_preview(x0, blocking)
        return ("JPEG", preview_image, MAX_PREVIEW_RESOLUTION)

    def decode_latent_to_preview(self, x0, blocking = False):
        self.snarf_all(self.pending)
        x0_slice = x0[:1].to(self.device)
        (result, _) = self.snarf_all(self.ready)
        try:
            self.pending.put_nowait(x0_slice)
        except queue.Full:
            print("warning: Lazy TAESD: Worker queue full, cannot submit")
        if result is None:
            if blocking:
                (result, _) = self.snarf_all(self.ready, True)
            else:
                return self.last_preview
        self.last_preview = result
        return self.last_preview


class TAESDPreviewerImpl(LatentPreviewer):
    def __init__(self, taesd):
        self.taesd = taesd

    def decode_latent_to_preview(self, x0):
        x_sample = self.taesd.decode(x0[:1])[0].detach()
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
        x_sample = x_sample.astype(np.uint8)

        preview_image = Image.fromarray(x_sample)
        return preview_image


class Latent2RGBPreviewer(LatentPreviewer):
    def __init__(self, latent_rgb_factors):
        self.latent_rgb_factors = torch.tensor(latent_rgb_factors, device="cpu")

    def decode_latent_to_preview(self, x0):
        latent_image = x0[0].permute(1, 2, 0).cpu() @ self.latent_rgb_factors

        latents_ubyte = (((latent_image + 1) / 2)
                            .clamp(0, 1)  # change scale from -1..1 to 0..1
                            .mul(0xFF)  # to 0..255
                            .byte()).cpu()

        return Image.fromarray(latents_ubyte.numpy())


def get_previewer(device, latent_format):
    previewer = None
    method = args.preview_method
    if method != LatentPreviewMethod.NoPreviews:
        # TODO previewer methods
        taesd_decoder_path = None
        if latent_format.taesd_decoder_name is not None:
            taesd_decoder_path = next(
                (fn for fn in folder_paths.get_filename_list("vae_approx")
                    if fn.startswith(latent_format.taesd_decoder_name)),
                ""
            )
            taesd_decoder_path = folder_paths.get_full_path("vae_approx", taesd_decoder_path)

        if method == LatentPreviewMethod.Auto:
            method = LatentPreviewMethod.Latent2RGB
            if taesd_decoder_path:
                method = LatentPreviewMethod.TAESD

        if method == LatentPreviewMethod.TAESD:
            if taesd_decoder_path:
                taesd = TAESD(None, taesd_decoder_path).to(device)
                previewer = LazyTAESDPreviewerImpl(taesd)
            else:
                print("Warning: TAESD previews enabled, but could not find models/vae_approx/{}".format(latent_format.taesd_decoder_name))

        if previewer is None:
            if latent_format.latent_rgb_factors is not None:
                previewer = Latent2RGBPreviewer(latent_format.latent_rgb_factors)
    return previewer

def prepare_callback(model, steps, x0_output_dict=None):
    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewer = get_previewer(model.load_device, model.model.latent_format)

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0

        preview_bytes = None
        if previewer:
            if isinstance(previewer, LazyTAESDPreviewerImpl) and step + 1 >= total_steps:
                # Wait for preview to complete on the last step.
                preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0, True)
            else:
                preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)
    return callback

