"""
Social Toolkit - Main entrance file for social media integration tools
"""

from . import SocialToolkitNode
from .architecture.document_retrieval_from_websites import DocumentRetrievalFromWebsites
from .architecture.document_storage import DocumentStorage
from .architecture.llm_service import LLMService
from .architecture.high_level_architecture import Socialtoolkit
from .architecture.codebook import Codebook

class SocialToolkitAPI:
    """API for accessing Social Toolkit functionality from other modules"""
    
    def __init__(self, resources, configs):
        self.configs = configs
        self.resources = resources

        self._document_retrieval_from_websites = self.resources["document_retrieval_from_websites"] or DocumentRetrievalFromWebsites(resources, configs)
        self._document_storage = self.resources["document_storage"] or DocumentStorage(resources, configs)
        self.llm_service = self.resources["llm_service"] or LLMService(resources, configs)
        self.control_flow = self.resources["socialtoolkit"] or Socialtoolkit(resources, configs)
        self.codebook = self.resources["codebook"] or Codebook(resources, configs)

    def document_retrieval_from_websites(self, domain_urls: list[str]) -> tuple['Document', 'Metadata', 'Vectors']:
        return self._document_retrieval_from_websites.execute(domain_urls)

    def document_storage(self):
        return self._document_storage.execute()

    def llm_service(self):
        pass

    def control_flow(self):
        pass


# Main function that can be called when using this as a script
def main():
    """Main function for Socialtoolkit module"""
    configs = Configs()
    resources = {
        "document_retrieval_from_websites": DocumentRetrievalFromWebsites(resources, configs),
        "document_storage": DocumentStorage(resources, configs),
        "llm_service": LLMService(resources, configs),
        "socialtoolkit": Socialtoolkit(resources, configs),
        "codebook": Codebook(resources, configs)
    }


    print("Social Toolkit module loaded successfully")
    print("Available tools:")
    print("- SocialToolkitNode: Node for ComfyUI integration")
    print("- SocialToolkitAPI: API for programmatic access")

if __name__ == "__main__":
    main()