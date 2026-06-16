import torch
from PIL import Image
from comfy.cli_args import args, LatentPreviewMethod
from comfy.taesd.taesd import TAESD
from comfy.sd import VAE
import comfy.model_management
import folder_paths
import comfy.utils
import logging

default_preview_method = args.preview_method

MAX_PREVIEW_RESOLUTION = args.preview_size
VIDEO_TAES = ["taehv", "lighttaew2_2", "lighttaew2_1", "lighttaehy1_5", "taeltx_2"]

def preview_to_image(latent_image, do_scale=True):
        if do_scale:
            latents_ubyte = (((latent_image + 1.0) / 2.0).clamp(0, 1)  # change scale from -1..1 to 0..1
                                .mul(0xFF)  # to 0..255
                                )
        else:
            latents_ubyte = (latent_image.clamp(0, 1)
                                .mul(0xFF)  # to 0..255
                                )
        if comfy.model_management.directml_enabled:
                latents_ubyte = latents_ubyte.to(dtype=torch.uint8)
        latents_ubyte = latents_ubyte.to(device="cpu", dtype=torch.uint8, non_blocking=comfy.model_management.device_supports_non_blocking(latent_image.device))

        return Image.fromarray(latents_ubyte.numpy())

class LatentPreviewer:
    def decode_latent_to_preview(self, x0):
        pass

    def decode_latent_to_preview_image(self, preview_format, x0):
        preview_image = self.decode_latent_to_preview(x0)
        return ("JPEG", preview_image, MAX_PREVIEW_RESOLUTION)

class TAESDPreviewerImpl(LatentPreviewer):
    def __init__(self, taesd):
        self.taesd = taesd

    def decode_latent_to_preview(self, x0):
        x_sample = self.taesd.decode(x0[:1])[0].movedim(0, 2)
        return preview_to_image(x_sample)

class TAEHVPreviewerImpl(TAESDPreviewerImpl):
    def decode_latent_to_preview(self, x0):
        x_sample = self.taesd.decode(x0[:1, :, :1])[0][0]
        return preview_to_image(x_sample, do_scale=False)

class Latent2RGBPreviewer(LatentPreviewer):
    def __init__(self, latent_rgb_factors, latent_rgb_factors_bias=None, latent_rgb_factors_reshape=None):
        self.latent_rgb_factors = torch.tensor(latent_rgb_factors, device="cpu").transpose(0, 1)
        self.latent_rgb_factors_bias = None
        if latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = torch.tensor(latent_rgb_factors_bias, device="cpu")
        self.latent_rgb_factors_reshape = latent_rgb_factors_reshape

    def decode_latent_to_preview(self, x0):
        if self.latent_rgb_factors_reshape is not None:
            x0 = self.latent_rgb_factors_reshape(x0)
        self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
        if self.latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(dtype=x0.dtype, device=x0.device)

        if x0.ndim == 5:
            x0 = x0[0, :, 0]
        else:
            x0 = x0[0]

        latent_image = torch.nn.functional.linear(x0.movedim(0, -1), self.latent_rgb_factors, bias=self.latent_rgb_factors_bias)
        # latent_image = x0[0].permute(1, 2, 0) @ self.latent_rgb_factors

        return preview_to_image(latent_image)


class Trellis3DPreviewer(LatentPreviewer):
    """Per-step preview for the Trellis2/Pixal3D cascade.

    Structure stage: x0 is a dense [B, 32, 16, 16, 16] grid — project the per-cell
    activation norm orthographically to a 2D occupancy heatmap (no decode, no coords).
    Texture stage: x0 is sparse [B, 32, N, 1] — splat the first 3 latent channels as
    pseudo-color onto the fixed voxel coords (read from the sampling side-channel).
    Shape stage adds no visible motion (coords are fixed, only sub-voxel detail
    evolves) and a full decode per step is too costly, so it's skipped.

    Both stages render through one orthographic point splatter (static view).
    """
    _SIZE = 128
    _FILL = 0.9                 # fraction of frame the texture splat spans (leaves a border)
    _STRUCTURE_ZOOM = 0.66      # <1 pulls the SS camera back, leaving margin around the blob

    def _splat(self, points, colors, rad):
        # points: [K, 3] voxel-index coords. colors: [K, 3] in [0, 1].
        # Center + isotropic-normalize, project orthographically front-on
        # (x->horizontal, y->up, z->depth), then splat a square footprint per point
        # with one global far->near sort (painter's).
        S = self._SIZE
        dev = points.device                                 # keep every tensor here
        p = points.float()
        p = p - (p.amax(0) + p.amin(0)) * 0.5
        p = p / p.abs().amax().clamp(min=1e-8)
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        depth = z                                           # into-screen
        m = self._FILL
        u = ((x * m * 0.5 + 0.5) * (S - 1)).long().clamp(0, S - 1)
        v = (((-y) * m * 0.5 + 0.5) * (S - 1)).long().clamp(0, S - 1)   # image up = +y
        cols = colors.to(dev)
        us, vs, ds, cs = [], [], [], []
        for dv in range(-rad, rad + 1):
            for du in range(-rad, rad + 1):
                us.append((u + du).clamp(0, S - 1))
                vs.append((v + dv).clamp(0, S - 1))
                ds.append(depth)
                cs.append(cols)
        order = torch.cat(ds).argsort()
        img = torch.zeros(S, S, 3, device=dev)
        img[torch.cat(vs)[order], torch.cat(us)[order]] = torch.cat(cs)[order]
        return preview_to_image(img, do_scale=False)

    @staticmethod
    def _turbo(x):
        # Anton Mikhailov polynomial approximation of the turbo colormap. x: any shape
        # in [0, 1] -> (..., 3) RGB.
        x = x.clamp(0.0, 1.0)
        x2 = x * x; x3 = x2 * x; x4 = x2 * x2; x5 = x4 * x
        r = 0.13572138 + 4.61539260*x - 42.66032258*x2 + 132.13108234*x3 - 152.94239396*x4 + 59.28637943*x5
        g = 0.09140261 + 2.19418839*x + 4.84296658*x2 - 14.18503333*x3 + 4.27729857*x4 + 2.82956604*x5
        b = 0.10667330 + 12.64194608*x - 60.58204836*x2 + 110.36276771*x3 - 89.90310912*x4 + 27.34824973*x5
        return torch.stack([r, g, b], dim=-1).clamp(0.0, 1.0)

    def _structure(self, x0):
        # x0: [B, 32, D, H, W]; the model only consumes the first 8 channels.
        # Dense orthographic max-projection -> filled occupancy heatmap (turbo-colored,
        # intensity-weighted so empty space stays black).
        act = x0[0, :min(8, x0.shape[1])].float().norm(dim=0)  # [D, H, W]
        proj = act.amax(dim=2)                                  # project along one axis
        proj = (proj - proj.amin()) / (proj.amax() - proj.amin() + 1e-8)
        inner = max(1, int(round(self._SIZE * self._STRUCTURE_ZOOM)))
        img = torch.nn.functional.interpolate(proj[None, None], size=(inner, inner), mode="nearest")
        pad = self._SIZE - inner
        pl, pt = pad // 2, pad // 2
        gray = torch.nn.functional.pad(img, (pl, pad - pl, pt, pad - pt))[0, 0]  # [S, S], zero margin
        rgb = self._turbo(gray) * gray.unsqueeze(-1)            # [S, S, 3], black where empty
        return preview_to_image(rgb, do_scale=False)

    @staticmethod
    def _latent_color(latent):
        # Prefer the calibrated latent->base_color map (fit from real decoded
        # albedo by VaeDecodeTextureTrellis); fall back to PCA pseudo-color until
        # a texture decode has trained it.
        try:
            from comfy.ldm.trellis2 import sampling_preview
            factors = sampling_preview.get_tex_rgb()
        except Exception:
            factors = None
        if factors is not None:
            W, b = factors
            rgb = latent @ W.to(latent) + b.to(latent)
            return rgb.clamp(0, 1)
        return Trellis3DPreviewer._pca_color(latent)

    @staticmethod
    def _pca_color(latent):
        # latent: [n, C]. Map the 3 directions of maximum variance to RGB.
        # Higher contrast and more coherent than picking 3 fixed channels.
        X = latent - latent.mean(dim=0, keepdim=True)
        cov = (X.transpose(0, 1) @ X) / max(X.shape[0] - 1, 1)   # [C, C]
        _, evecs = torch.linalg.eigh(cov)                         # ascending eigenvalues
        pcs = evecs[:, -3:]                                       # [C, 3] top-3 components
        # Deterministic sign per component (largest-magnitude entry positive) to
        # stop the preview's hues from flickering as the latent rotates each step.
        sign = torch.sign(pcs[pcs.abs().argmax(dim=0), torch.arange(3, device=pcs.device)])
        pcs = pcs * sign.clamp(min=-1.0)
        proj = X @ pcs                                            # [n, 3]
        pmin = proj.amin(dim=0, keepdim=True)
        pmax = proj.amax(dim=0, keepdim=True)
        return ((proj - pmin) / (pmax - pmin + 1e-8)).clamp(0, 1)

    def _texture(self, x0, coords, model_frame=None):
        if coords.shape[-1] == 4:
            b0 = coords[:, 0] == 0
            spatial = coords[b0][:, 1:4].float()
        else:
            spatial = coords[:, :3].float()
        n0 = spatial.shape[0]
        if n0 == 0:
            return None
        if model_frame == "z_up":
            spatial = torch.stack([spatial[:, 0], spatial[:, 2], -spatial[:, 1]], dim=-1)
        latent = x0[0, :, :n0, 0].float().transpose(0, 1)        # [n0, C]
        colors = self._latent_color(latent)                      # [n0, 3]
        res = float(spatial.abs().max().item()) + 1.0
        rad = max(1, int(round(self._SIZE * self._FILL / max(res, 1) / 2)))
        return self._splat(spatial, colors, rad)

    def decode_latent_to_preview(self, x0):
        try:
            from comfy.ldm.trellis2 import sampling_preview
            ctx = sampling_preview.get_context()
            if x0.ndim == 5:
                return self._structure(x0)
            mode = ctx.get("mode")
            coords = ctx.get("coords")
            if mode == "texture_generation" and coords is not None:
                return self._texture(x0, coords, model_frame=ctx.get("model_frame"))
        except Exception as e:
            logging.debug(f"Trellis3DPreviewer: skipping preview ({e})")
        return None

    def decode_latent_to_preview_image(self, preview_format, x0):
        preview_image = self.decode_latent_to_preview(x0)
        if preview_image is None:
            return None
        return ("JPEG", preview_image, MAX_PREVIEW_RESOLUTION)


def get_previewer(device, latent_format):
    previewer = None
    method = args.preview_method
    if method != LatentPreviewMethod.NoPreviews:
        if getattr(latent_format, "trellis3d_preview", False):
            return Trellis3DPreviewer()
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

        if method == LatentPreviewMethod.TAESD:
            if taesd_decoder_path:
                if latent_format.taesd_decoder_name in VIDEO_TAES:
                    taesd = VAE(comfy.utils.load_torch_file(taesd_decoder_path))
                    taesd.first_stage_model.show_progress_bar = False
                    previewer = TAEHVPreviewerImpl(taesd)
                else:
                    taesd = TAESD(None, taesd_decoder_path, latent_channels=latent_format.latent_channels).to(device)
                    previewer = TAESDPreviewerImpl(taesd)
            else:
                logging.warning("Warning: TAESD previews enabled, but could not find models/vae_approx/{}".format(latent_format.taesd_decoder_name))

        if previewer is None:
            if latent_format.latent_rgb_factors is not None:
                previewer = Latent2RGBPreviewer(latent_format.latent_rgb_factors, latent_format.latent_rgb_factors_bias, latent_format.latent_rgb_factors_reshape)
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
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)
    return callback

def set_preview_method(override: str = None):
    if override and override != "default":
        method = LatentPreviewMethod.from_string(override)
        if method is not None:
            args.preview_method = method
            return
    args.preview_method = default_preview_method

