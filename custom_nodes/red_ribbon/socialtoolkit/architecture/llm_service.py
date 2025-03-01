from logging import Logger
from typing import Any

from configs import Configs

class LLMService:

    def __init__(self, resources: dict[str, Any], configs: Configs):
        self.resources = resources
        self.configs = configs
        self.logger: Logger = resources.get("logger")

        self.llm_model = resources.get("llm_model")
        self.llm_tokenizer = resources.get("llm_tokenizer")
        self.llm_vectorizer = resources.get("llm_vectorizer")
        self.llm_vector_storage = resources.get("llm_vector_storage")

        self.logger.info("LLMService initialized with services")

    def execute(self, command_context: str, *args, **kwargs):
        """
        
        
        """
        pass


