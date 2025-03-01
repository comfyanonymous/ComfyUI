"""
Red Ribbon - Main module for importing and registering all nodes
"""
from typing import Type


from easy_nodes import (
    NumberInput,
    ComfyNode,
    StringInput,
    Choice,
)


# Import components from subdirectories
from .socialtoolkit.socialtoolkit import SocialToolkitAPI
from .red_ribbon_core.red_ribbon import RedRibbonAPI
from .plug_in_play_transformer.plug_in_play_transformer import TransformerAPI
from .utils.utils import UtilsAPI
from .configs import Configs
from .node_types import register_pydantic_models


modules_to_register = [
    "red_ribbon",
    "socialtoolkit",
    "utils",
    "plug_in_play_transformer",
    "configs",
]
register_pydantic_models(modules_to_register)


class RedRibbonPackage:
    """Main interface for the Red Ribbon package"""
    
    def __init__(self, resources: dict[str, object] = None, configs: Configs = None):
        """Initialize the Red Ribbon package components"""
        self.configs = configs
        self.resources = resources

        self.social: Type[SocialToolkitAPI] = self.resources.get("social")
        self.rr: Type[RedRibbonAPI] = self.resources.get("rr")
        self.trans: Type[TransformerAPI] = self.resources.get("trans")
        self.utils: Type[UtilsAPI] = self.resources.get("utils")

    def version(self):
        """Get the version of the Red Ribbon package"""
        from . import __version__
        return __version__


rr_resources = {
}
social_resources = {
}
trans_resources = {
}
utils_resources = {
}

configs = Configs()
resources = {
    "social": SocialToolkitAPI(social_resources, configs),
    "rr": RedRibbonAPI(rr_resources, configs),
    "trans": TransformerAPI(trans_resources, configs),
    "utils": UtilsAPI(utils_resources, configs)
}

# Initialize the Red Ribbon package
package = RedRibbonPackage(resources, configs)


@ComfyNode("Socialtoolkit",
           color="#d30e0e",
           bg_color="#ff0000",
           display_name="Rank and Sort Similarity Search Results")
def rank_and_sort_similar_search_results(
    search_results: list,
    search_query: str,
    search_type: str,
    rank_by: str,
    sort_by: str
) -> list:
    """
    Rank and sort similarity search results.
    """
    return package.social.rank_and_sort_similar_search_results(
        search_results,
        search_query,
        search_type,
        rank_by,
        sort_by
    )

@ComfyNode("Socialtoolkit",
           color="#d30e0e",
           bg_color="#ff0000",
           display_name="Retrieve Documents from Websites")
def document_retrieval_from_websites(
    domain_urls: list[str]
) -> tuple['Document', 'Metadata', 'Vectors']:
    """
    Document retrieval from websites.
    """
    resources: dict[str, object],
    configs: Configs

    socialtoolkit = SocialToolkitAPI(resources, configs)
    return socialtoolkit.document_retrieval_from_websites(
        domain_urls
    )


# Main function that can be called when using this as a script
def main():
    print("Red Ribbon package loaded successfully")
    package = RedRibbonPackage()
    print(f"Version: {package.version()}")
    print("Available components:")
    print("- SocialToolkit")
    print("- RedRibbon Core")
    print("- Plug-in-Play Transformer")
    print("- Utils")

if __name__ == "__main__":
    main()


