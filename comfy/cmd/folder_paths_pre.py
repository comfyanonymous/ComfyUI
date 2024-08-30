import logging
import os

from ..cli_args import args

_base_path = None


# todo: this should be initialized elsewhere in a context
def get_base_path() -> str:
    global _base_path
    if _base_path is None:
        if args.cwd is not None:
            if not os.path.exists(args.cwd):
                try:
                    os.makedirs(args.cwd, exist_ok=True)
                except:
                    logging.error("Failed to create custom working directory")
                # wrap the path to prevent slashedness from glitching out common path checks
            _base_path = os.path.realpath(args.cwd)
        else:
            _base_path = os.getcwd()
    return _base_path


def set_base_path(value: str):
    global _base_path
    _base_path = value


__all__ = ["get_base_path", "set_base_path"]
