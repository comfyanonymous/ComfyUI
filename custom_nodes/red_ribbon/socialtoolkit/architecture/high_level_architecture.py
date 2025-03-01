from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Type
import logging

logger = logging.getLogger(__name__)

from configs import Configs


class SocialtoolkitConfigs(BaseModel):
    """Configuration for High Level Architecture workflow"""
    approved_document_sources: List[str]
    llm_api_config: Dict[str, Any]
    document_retrieval_threshold: int = 10
    relevance_threshold: float = 0.7
    output_format: str = "json"
    get_documents_from_web: bool = False


class Socialtoolkit:
    """
    High Level Architecture for document retrieval and data extraction system
    based on mermaid chart in README.md
    """
    
    def __init__(self, resources: Dict[str, Any], configs: Configs):
        """
        Initialize with injected dependencies and configuration
        
        Args:
            resources: Dictionary of resources including services
            configs: Configuration for High Level Architecture
        """
        self.resources = resources
        self.configs: SocialtoolkitConfigs = configs.socialtoolkit
        self.llm_api = self.llm_service(resources, configs)
        
        # Extract needed services from resources
        self.document_retrieval = resources.get("document_retrieval_service")
        self.document_storage = resources.get("document_storage_service")
        self.llm_service = resources.get("llm_service")
        self.top10_retrieval = resources.get("top10_retrieval_service")
        self.relevance_assessment = resources.get("relevance_assessment_service")
        self.prompt_decision_tree = resources.get("prompt_decision_tree_service")
        self.variable_codebook = resources.get("variable_codebook_service")
        
        logger.info("Socialtoolkit initialized with services")


    def execute(self, input_data_point: str) -> dict[str, str] | list[dict[str, str]]:
        """
        Execute the control flow based on the mermaid chart
        
        Args:
            input_data_point: The question or information request. This can be a single request.

        Returns:
            Dictionary containing the output data point. 
            If the request was interpreted as having more than one response, a list of dictionaries is returned.
        """
        logger.info(f"Starting high level control flow with input: {input_data_point}")
        
        if self.configs.approved_document_sources:
            # Step 1: Get domain URLs from pre-approved sources
            domain_urls: list[str] = self.document_retrieval.execute(domain_urls)
            
            # Step 2: Retrieve documents from websites
            documents, metadata, vectors = self.document_retrieval.execute(domain_urls)
            documents: list[tuple[str, ...]]
            metadata: list[dict[str, Any]]
            vectors: list[dict[str, list[float]]]
            
            # Step 3: Store documents in document storage
            storage_successful: bool = self.document_storage.execute(documents, metadata, vectors)
            if storage_successful:
                logger.info("Documents stored successfully")
            else:
                logger.warning("Failed to store documents")
        
        # Step 4: Retrieve documents and document vectors
        stored_docs, stored_vectors = self.document_retrieval.execute(
            input_data_point,
            self.llm_service.execute("retrieve_documents")
        )
        stored_docs: list[tuple[str, ...]]
        stored_vectors: list[dict[str, list[float]]]
        
        # Step 5: Perform top-10 document retrieval
        potentially_relevant_docs = self.top10_retrieval.execute(
            input_data_point, 
            stored_docs, 
            stored_vectors
        )
        potentially_relevant_docs: list[tuple[str, ...]]
        
        # Step 6: Get variable definition from codebook
        prompt_sequence = self.variable_codebook.execute(self.llm_service, input_data_point)
        
        # Step 7: Perform relevance assessment
        relevant_documents = self.relevance_assessment.execute(
            potentially_relevant_docs,
            prompt_sequence,
            self.llm_service.execute("relevance_assessment")
        )
        
        # Step 8: Execute prompt decision tree
        output_data_point = self.prompt_decision_tree.execute(
            relevant_documents,
            prompt_sequence,
            self.llm_service.execute("prompt_decision_tree")
        )

        if output_data_point is None:
            logger.warning("Failed to execute prompt decision tree")
        else:
            logger.info(f"Completed high level control flow with output: {output_data_point}")
        
        return {"output_data_point": output_data_point}
        
    def _get_domain_urls(self) -> List[str]:
        """Extract domain URLs from pre-approved document sources"""
        return self.configs.approved_document_sources