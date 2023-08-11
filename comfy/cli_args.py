import argparse
import enum


class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums
    """
    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        choices = tuple(e.value for e in enum_type)
        kwargs.setdefault("choices", choices)
        kwargs.setdefault("metavar", f"[{','.join(list(choices))}]")

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum(values)
        setattr(namespace, self.dest, value)


parser = argparse.ArgumentParser()

parser.add_argument("--listen", type=str, default="127.0.0.1", metavar="IP", nargs="?", const="0.0.0.0", help="Specify the IP address to listen on (default: 127.0.0.1). If --listen is provided without an argument, it defaults to 0.0.0.0. (listens on all)")
parser.add_argument("--port", type=int, default=8188, help="Set the listen port.")
parser.add_argument("--enable-cors-header", type=str, default=None, metavar="ORIGIN", nargs="?", const="*", help="Enable CORS (Cross-Origin Resource Sharing) with optional origin or allow all with default '*'.")
parser.add_argument("--extra-model-paths-config", type=str, default=None, metavar="PATH", nargs='+', action='append', help="Load one or more extra_model_paths.yaml files.")
parser.add_argument("--output-directory", type=str, default=None, help="Set the ComfyUI output directory.")
parser.add_argument("--temp-directory", type=str, default=None, help="Set the ComfyUI temp directory (default is in the ComfyUI directory).")
parser.add_argument("--auto-launch", action="store_true", help="Automatically launch ComfyUI in the default browser.")
parser.add_argument("--disable-auto-launch", action="store_true", help="Disable auto launching the browser.")
parser.add_argument("--cuda-device", type=int, default=None, metavar="DEVICE_ID", help="Set the id of the cuda device this instance will use.")
cm_group = parser.add_mutually_exclusive_group()
cm_group.add_argument("--cuda-malloc", action="store_true", help="Enable cudaMallocAsync (enabled by default for torch 2.0 and up).")
cm_group.add_argument("--disable-cuda-malloc", action="store_true", help="Disable cudaMallocAsync.")

parser.add_argument("--dont-upcast-attention", action="store_true", help="Disable upcasting of attention. Can boost speed but increase the chances of black images.")

fp_group = parser.add_mutually_exclusive_group()
fp_group.add_argument("--force-fp32", action="store_true", help="Force fp32 (If this makes your GPU work better please report it).")
fp_group.add_argument("--force-fp16", action="store_true", help="Force fp16.")

fpvae_group = parser.add_mutually_exclusive_group()
fpvae_group.add_argument("--fp16-vae", action="store_true", help="Run the VAE in fp16, might cause black images.")
fpvae_group.add_argument("--bf16-vae", action="store_true", help="Run the VAE in bf16, might lower quality.")

parser.add_argument("--directml", type=int, nargs="?", metavar="DIRECTML_DEVICE", const=-1, help="Use torch-directml.")

class LatentPreviewMethod(enum.Enum):
    NoPreviews = "none"
    Auto = "auto"
    Latent2RGB = "latent2rgb"
    TAESD = "taesd"

parser.add_argument("--preview-method", type=LatentPreviewMethod, default=LatentPreviewMethod.NoPreviews, help="Default preview method for sampler nodes.", action=EnumAction)

attn_group = parser.add_mutually_exclusive_group()
attn_group.add_argument("--use-split-cross-attention", action="store_true", help="Use the split cross attention optimization. Ignored when xformers is used.")
attn_group.add_argument("--use-quad-cross-attention", action="store_true", help="Use the sub-quadratic cross attention optimization . Ignored when xformers is used.")
attn_group.add_argument("--use-pytorch-cross-attention", action="store_true", help="Use the new pytorch 2.0 cross attention function.")

parser.add_argument("--disable-xformers", action="store_true", help="Disable xformers.")

vram_group = parser.add_mutually_exclusive_group()
vram_group.add_argument("--gpu-only", action="store_true", help="Store and run everything (text encoders/CLIP models, etc... on the GPU).")
vram_group.add_argument("--highvram", action="store_true", help="By default models will be unloaded to CPU memory after being used. This option keeps them in GPU memory.")
vram_group.add_argument("--normalvram", action="store_true", help="Used to force normal vram use if lowvram gets automatically enabled.")
vram_group.add_argument("--lowvram", action="store_true", help="Split the unet in parts to use less vram.")
vram_group.add_argument("--novram", action="store_true", help="When lowvram isn't enough.")
vram_group.add_argument("--cpu", action="store_true", help="To use the CPU for everything (slow).")


parser.add_argument("--dont-print-server", action="store_true", help="Don't print server output.")
parser.add_argument("--quick-test-for-ci", action="store_true", help="Quick test for CI.")
parser.add_argument("--windows-standalone-build", action="store_true", help="Windows standalone build: Enable convenient things that most people using the standalone windows build will probably enjoy (like auto opening the page on startup).")

parser.add_argument("--disable-metadata", action="store_true", help="Disable saving prompt metadata in files.")

args = parser.parse_args()

if args.windows_standalone_build:
    args.auto_launch = True

if args.disable_auto_launch:
    args.auto_launch = False
