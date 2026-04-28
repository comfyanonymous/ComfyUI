# pylint: disable=import-outside-toplevel
# Host process initialization for PyIsolate
import logging

logger = logging.getLogger(__name__)


def initialize_host_process() -> None:
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    root.addHandler(logging.NullHandler())

    from .proxies.folder_paths_proxy import FolderPathsProxy
    from .proxies.helper_proxies import HelperProxiesService
    from .proxies.model_management_proxy import ModelManagementProxy
    from .proxies.progress_proxy import ProgressProxy
    from .proxies.prompt_server_impl import PromptServerService
    from .proxies.utils_proxy import UtilsProxy
    from .proxies.web_directory_proxy import WebDirectoryProxy
    from .vae_proxy import VAERegistry

    FolderPathsProxy()
    HelperProxiesService()
    ModelManagementProxy()
    ProgressProxy()
    PromptServerService()
    UtilsProxy()
    WebDirectoryProxy()
    VAERegistry()
