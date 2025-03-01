from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

class Top10DocumentRetrievalConfigs(BaseModel):
    """Configuration for Top-10 Document Retrieval workflow"""
    retrieval_count: int = 10  # Number of documents to retrieve
    similarity_threshold: float = 0.6  # Minimum similarity score
    ranking_method: str = "cosine_similarity"  # Options: cosine_similarity, dot_product, euclidean
    use_filter: bool = False  # Whether to filter results
    filter_criteria: Dict[str, Any] = {}
    use_reranking: bool = False  # Whether to use reranking

class Top10DocumentRetrieval:
    """
    Top-10 Document Retrieval system based on mermaid chart in README.md
    Performs vector search to find the most relevant documents
    """
    
    def __init__(self, resources: Dict[str, Any], configs: Top10DocumentRetrievalConfigs):
        """
        Initialize with injected dependencies and configuration
        
        Args:
            resources: Dictionary of resources including search services
            configs: Configuration for Top-10 Document Retrieval
        """
        self.resources = resources
        self.configs = configs
        
        # Extract needed services from resources
        self.encoder_service = resources.get("encoder_service")
        self.similarity_search_service = resources.get("similarity_search_service")
        self.document_storage = resources.get("document_storage_service")
        
        logger.info("Top10DocumentRetrieval initialized with services")

    def execute(self, 
                    input_data_point: str, 
                    documents: List[Any] = None, 
                    document_vectors: List[Any] = None
                   ) -> dict[str, Any]:
        """
        Execute the document retrieval flow based on the mermaid chart
        
        Args:
            input_data_point: The query or information request
            documents: Optional list of documents to search
            document_vectors: Optional list of document vectors to search
            
        Returns:
            Dictionary of documents containing potentially relevant documents, along with potentially relevant metadata.
        """
        logger.info(f"Starting top-10 document retrieval for: {input_data_point}")
        
        # Step 1: Encode the query
        encoded_query = self._encode_query(input_data_point)
        
        # Step 2: Get vector embeddings and document IDs from storage if not provided
        if documents is None or document_vectors is None:
            documents, document_vectors = self._get_documents_and_vectors()
        
        # Step 3: Perform similarity search
        similarity_scores, doc_ids = self._similarity_search(
            encoded_query, 
            document_vectors, 
            [doc.get("id") for doc in documents]
        )
        
        # Step 4: Rank and sort results
        ranked_results = self._rank_and_sort_results(similarity_scores, doc_ids)
        
        # Step 5: Filter to top-N results
        top_doc_ids = self._filter_to_top_n(ranked_results)
        
        # Step 6: Retrieve potentially relevant documents
        potentially_relevant_docs = self._retrieve_relevant_documents(documents, top_doc_ids)
        
        logger.info(f"Retrieved {len(potentially_relevant_docs)} potentially relevant documents")
        return {
            "relevant_documents": potentially_relevant_docs,
            "scores": {doc_id: score for doc_id, score in ranked_results},
            "top_doc_ids": top_doc_ids
        }

    def retrieve_top_documents(self, input_data_point: str, documents: List[Any], document_vectors: List[Any]) -> List[Any]:
        """
        Public method to retrieve top documents for an input query
        
        Args:
            input_data_point: The query to search for
            documents: Documents to search
            document_vectors: Vectors for the documents
            
        Returns:
            List of potentially relevant documents
        """
        result = self.control_flow(input_data_point, documents, document_vectors)
        return result["relevant_documents"]
    
    def _encode_query(self, input_data_point: str) -> Any:
        """
        Encode the input query into a vector representation
        
        Args:
            input_data_point: The query to encode
            
        Returns:
            Vector representation of the query
        """
        logger.debug(f"Encoding query: {input_data_point}")
        return self.encoder_service.encode(input_data_point)
    
    def _get_documents_and_vectors(self) -> Tuple[List[Any], List[Any]]:
        """
        Get all documents and their vectors from storage
        
        Returns:
            Tuple of (documents, document_vectors)
        """
        logger.debug("Getting documents and vectors from storage")
        return self.document_storage.get_documents_and_vectors()
    
    def _similarity_search(self, encoded_query: Any, document_vectors: List[Any], 
                          doc_ids: List[str]) -> Tuple[List[float], List[str]]:
        """
        Perform similarity search between the query and document vectors
        
        Args:
            encoded_query: Vector representation of the query
            document_vectors: List of document vector embeddings
            doc_ids: List of document IDs corresponding to the vectors
            
        Returns:
            Tuple of (similarity_scores, document_ids)
        """
        logger.debug("Performing similarity search")
        
        # In a real implementation, this would use an efficient vector search
        similarity_scores = []
        
        for vector in document_vectors:
            if self.configs.ranking_method == "cosine_similarity":
                score = self._cosine_similarity(encoded_query, vector.get("embedding"))
            elif self.configs.ranking_method == "dot_product":
                score = self._dot_product(encoded_query, vector.get("embedding"))
            elif self.configs.ranking_method == "euclidean":
                score = self._euclidean_distance(encoded_query, vector.get("embedding"))
                # Convert distance to similarity score (higher is more similar)
                score = 1.0 / (1.0 + score)
            else:
                score = 0.0
                
            similarity_scores.append(score)
        
        # If the similarity search service is available, use it instead
        if self.similarity_search_service:
            return self.similarity_search_service.search(
                encoded_query, document_vectors, doc_ids
            )
            
        return similarity_scores, doc_ids
    
    def _rank_and_sort_results(self, similarity_scores: List[float], 
                              doc_ids: List[str]) -> List[Tuple[str, float]]:
        """
        Rank and sort results by similarity score
        
        Args:
            similarity_scores: List of similarity scores
            doc_ids: List of document IDs
            
        Returns:
            List of (document_id, score) tuples sorted by score
        """
        logger.debug("Ranking and sorting results")
        
        # Create a list of (document_id, score) tuples
        result_tuples = list(zip(doc_ids, similarity_scores))
        
        # Sort by score in descending order
        sorted_results = sorted(result_tuples, key=lambda x: x[1], reverse=True)
        
        return sorted_results
    
    def _filter_to_top_n(self, ranked_results: List[Tuple[str, float]]) -> List[str]:
        """
        Filter to top N results
        
        Args:
            ranked_results: List of (document_id, score) tuples
            
        Returns:
            List of top N document IDs
        """
        logger.debug(f"Filtering to top {self.configs.retrieval_count} results")
        
        # Apply threshold filter if configured
        filtered_results = []
        
        if self.configs.use_filter:
            for doc_id, score in ranked_results:
                if score >= self.configs.similarity_threshold:
                    filtered_results.append(doc_id)
        else:
            filtered_results = [doc_id for doc_id, _ in ranked_results]
        
        # Return top N results
        return filtered_results[:self.configs.retrieval_count]
    
    def _retrieve_relevant_documents(self, documents: List[Any], top_doc_ids: List[str]) -> List[Any]:
        """
        Retrieve potentially relevant documents
        
        Args:
            documents: List of all documents
            top_doc_ids: List of top document IDs
            
        Returns:
            List of potentially relevant documents
        """
        logger.debug("Retrieving potentially relevant documents")
        
        # Create a map of document ID to document for faster lookup
        doc_map = {doc.get("id"): doc for doc in documents}
        
        # Retrieve documents by ID
        relevant_docs = []
        
        for doc_id in top_doc_ids:
            if doc_id in doc_map:
                relevant_docs.append(doc_map[doc_id])
        
        return relevant_docs
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
            
        try:
            # Convert to numpy arrays for efficient calculation
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            
            # Calculate dot product
            dot = np.dot(vec1_np, vec2_np)
            
            # Calculate norms
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)
            
            # Calculate cosine similarity
            similarity = dot / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
            
    def _dot_product(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate dot product between two vectors"""
        if not vec1 or not vec2:
            return 0.0
            
        try:
            # Convert to numpy arrays for efficient calculation
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            
            # Calculate dot product
            dot = np.dot(vec1_np, vec2_np)
            return float(dot)
        except Exception as e:
            logger.error(f"Error calculating dot product: {e}")
            return 0.0
            
    def _euclidean_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate Euclidean distance between two vectors"""
        if not vec1 or not vec2:
            return float('inf')
            
        try:
            # Convert to numpy arrays for efficient calculation
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(vec1_np - vec2_np)
            return float(distance)
        except Exception as e:
            logger.error(f"Error calculating Euclidean distance: {e}")
            return float('inf')