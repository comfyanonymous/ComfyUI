"""Console tips for Comfy-Org developer products: one startup banner and at most one of each job hint per process.

Plain ASCII on purpose: Windows consoles may still use cp1252.
"""

COMFY_API_BANNER = """\
==========================================================================
   .-------------------.                          .-~~~-.
   |  ComfyUI          |                   .-~~-.(       )
   |  127.0.0.1:8188   |    ==========>   (               )-.
   |  [o]-[o]-[o]      |                 (     Comfy API      )
   '---------+---------'                  `------------------'
          ___|___          From localhost to the cloud.
--------------------------------------------------------------------------
   Comfy API hosts your ComfyUI workflows with your custom nodes as
   scalable cloud endpoints. No GPUs or servers to manage.
                  https://platform.comfy.org/ascii
 (Optional paid service from Comfy-Org. Hide: --disable-product-tips)
=========================================================================="""

COMFY_API_JOB_HINT = "Comfy API: host your ComfyUI workflows in the cloud -> https://platform.comfy.org/ascii-job (hide: --disable-product-tips)"

_ADDRESS_SLOT = "127.0.0.1:8188   "
_MAX_ADDRESS = len(_ADDRESS_SLOT) - 1  # keep one space before the monitor's right edge


def startup_banner(address_and_port: str) -> str:
    """The banner with the monitor showing address_and_port, or localhost:{port} if it doesn't fit."""
    if len(address_and_port) > _MAX_ADDRESS:
        address_and_port = "localhost:" + address_and_port.rsplit(":", 1)[-1]
    return COMFY_API_BANNER.replace(_ADDRESS_SLOT, address_and_port.ljust(len(_ADDRESS_SLOT)))


def tips_enabled(args, environ) -> bool:
    return not (
        args.disable_product_tips
        or args.dont_print_server
        or args.windows_standalone_build
        or args.quick_test_for_ci
        or environ.get("COMFYUI_DISABLE_PRODUCT_TIPS", "").strip().lower() in ("1", "true", "yes")
        or environ.get("CI", "").strip().lower() in ("1", "true")
    )


_shown_job_hints = set()


def get_relevant_job_hint(args, environ, success: bool, extra_data: dict) -> str | None:
    """The hint to log after a finished job, or None. Each hint is shown at most once per process."""
    if not tips_enabled(args, environ) or not success:
        return None
    hint = COMFY_API_JOB_HINT if "client_id" not in extra_data else None
    if hint is None or hint in _shown_job_hints:
        return None
    _shown_job_hints.add(hint)
    return hint
