from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class RelevanceAssessmentConfigs(BaseModel):
    """Configuration for Relevance Assessment workflow"""
    criteria_threshold: float = 0.7  # Minimum relevance score threshold
    max_retries: int = 3  # Maximum number of retry attempts for LLM API calls
    max_citation_length: int = 500  # Maximum length of text citations
    use_hallucination_filter: bool = True  # Whether to filter for hallucinations

class RelevanceAssessment:
    """
    Relevance Assessment system based on mermaid flowchart in README.md
    Evaluates document relevance using LLM assessments
    """
    
    def __init__(self, resources: Dict[str, Any], configs: RelevanceAssessmentConfigs):
        """
        Initialize with injected dependencies and configuration
        
        Args:
            resources: Dictionary of resources including services
            configs: Configuration for Relevance Assessment
        """
        self.resources = resources
        self.configs = configs
        
        # Extract needed services from resources
        self.variable_codebook = resources.get("variable_codebook_service")
        self.top10_retrieval = resources.get("top10_retrieval_service")
        self.cited_page_extractor = resources.get("cited_page_extractor_service")
        self.prompt_decision_tree = resources.get("prompt_decision_tree_service")
        
        logger.info("RelevanceAssessment initialized with services")

    def control_flow(self, potentially_relevant_docs: List[Any], 
                   variable_definition: Dict[str, Any],
                   llm_api: Any) -> Dict[str, Any]:
        """
        Execute the relevance assessment flow based on the mermaid flowchart
        
        Args:
            potentially_relevant_docs: List of potentially relevant documents
            variable_definition: Variable definition and description
            llm_api: LLM API instance
            
        Returns:
            Dictionary containing relevant documents and page numbers
        """
        logger.info(f"Starting relevance assessment for {len(potentially_relevant_docs)} documents")
        
        # Step 1: Assess document relevance
        assessment_results = self._assess_document_relevance(
            potentially_relevant_docs, 
            variable_definition, 
            llm_api
        )
        
        # Step 2: Filter for hallucinations if configured
        if self.configs.use_hallucination_filter:
            assessment_results = self._filter_hallucinations(assessment_results, llm_api)
        
        # Step 3: Score relevance
        relevance_scores = self._score_relevance(assessment_results, potentially_relevant_docs)
        
        # Step 4: Apply threshold to separate relevant from irrelevant
        relevant_pages, discarded_pages = self._apply_threshold(relevance_scores)
        
        # Step 5: Extract page numbers
        page_numbers = self._extract_page_numbers(relevant_pages)
        
        # Step 6: Extract cited pages
        relevant_pages_content = self._extract_cited_pages(
            potentially_relevant_docs, page_numbers
        )
        
        logger.info(f"Completed relevance assessment: {len(relevant_pages_content)} relevant pages")
        return {
            "relevant_pages": relevant_pages_content,
            "relevant_doc_ids": [page["doc_id"] for page in relevant_pages],
            "page_numbers": page_numbers,
            "relevance_scores": relevance_scores
        }

    def assess(self, potentially_relevant_docs: List[Any], 
              prompt_sequence: List[str], llm_api: Any) -> List[Any]:
        """
        Public method to assess document relevance
        
        Args:
            potentially_relevant_docs: List of potentially relevant documents
            prompt_sequence: List of prompts to use for assessment
            llm_api: LLM API instance
            
        Returns:
            List of relevant documents
        """
        # Get variable definition from prompt sequence
        variable_definition = {
            "prompt_sequence": prompt_sequence,
            "description": "Tax information for business operations"  # Default description if not available
        }
        
        result = self.control_flow(potentially_relevant_docs, variable_definition, llm_api)
        return result["relevant_pages"]
    
    def _assess_document_relevance(self, docs: List[Any], 
                                 variable_definition: Dict[str, Any], 
                                 llm_api: Any) -> List[Dict[str, Any]]:
        """
        Assess document relevance using LLM
        
        Args:
            docs: List of documents to assess
            variable_definition: Variable definition and description
            llm_api: LLM API instance
            
        Returns:
            List of assessment results
        """
        assessment_results = []
        
        for doc in docs:
            # Create assessment prompt
            assessment_prompt = self._create_assessment_prompt(doc, variable_definition)
            
            # Get LLM assessment
            try:
                llm_response = llm_api.generate(assessment_prompt, max_tokens=1000)
                
                # Parse assessment results
                assessment = self._parse_assessment(llm_response, doc)
                assessment_results.append(assessment)
                
            except Exception as e:
                logger.error(f"Error assessing document {doc.get('id')}: {e}")
                # Add failed assessment
                assessment_results.append({
                    "doc_id": doc.get("id"),
                    "relevant": False,
                    "confidence": 0.0,
                    "citation": "",
                    "error": str(e)
                })
        
        return assessment_results
    
    def _filter_hallucinations(self, assessments: List[Dict[str, Any]], 
                             llm_api: Any) -> List[Dict[str, Any]]:
        """
        Filter for hallucinations in LLM assessments
        
        Args:
            assessments: List of assessment results
            llm_api: LLM API instance
            
        Returns:
            Filtered list of assessment results
        """
        filtered_assessments = []
        
        for assessment in assessments:
            # Skip already irrelevant assessments
            if not assessment.get("relevant", False):
                filtered_assessments.append(assessment)
                continue
                
            # Create hallucination check prompt
            hallucination_prompt = self._create_hallucination_prompt(assessment)
            
            try:
                # Get LLM hallucination check
                hallucination_response = llm_api.generate(hallucination_prompt, max_tokens=500)
                
                # Parse hallucination check
                is_hallucination = self._parse_hallucination_check(hallucination_response)
                
                if is_hallucination:
                    # Downgrade relevance for hallucinations
                    assessment["relevant"] = False
                    assessment["confidence"] = 0.0
                    assessment["hallucination"] = True
                
                filtered_assessments.append(assessment)
                
            except Exception as e:
                logger.error(f"Error checking hallucination for document {assessment.get('doc_id')}: {e}")
                # Keep original assessment in case of error
                filtered_assessments.append(assessment)
        
        return filtered_assessments
    
    def _score_relevance(self, assessments: List[Dict[str, Any]], 
                       docs: List[Any]) -> List[Dict[str, Any]]:
        """
        Score relevance based on LLM assessments
        
        Args:
            assessments: List of assessment results
            docs: List of original documents
            
        Returns:
            List of documents with relevance scores
        """
        # Create a dict mapping document ID to original document
        doc_map = {doc.get("id"): doc for doc in docs}
        
        relevance_scores = []
        
        for assessment in assessments:
            doc_id = assessment.get("doc_id")
            doc = doc_map.get(doc_id)
            
            if not doc:
                logger.warning(f"Document not found for ID: {doc_id}")
                continue
                
            # Calculate relevance score based on LLM confidence
            relevance_score = {
                "doc_id": doc_id,
                "page_number": assessment.get("page_number", 1),  # Default to page 1 if not specified
                "score": assessment.get("confidence", 0.0),
                "relevant": assessment.get("relevant", False),
                "citation": assessment.get("citation", ""),
                "content": doc.get("content", "")
            }
            
            relevance_scores.append(relevance_score)
        
        return relevance_scores
    
    def _apply_threshold(self, relevance_scores: List[Dict[str, Any]]) -> tuple:
        """
        Apply threshold to relevance scores
        
        Args:
            relevance_scores: List of documents with relevance scores
            
        Returns:
            Tuple of (relevant_pages, discarded_pages)
        """
        relevant_pages = []
        discarded_pages = []
        
        for score in relevance_scores:
            if score.get("score", 0.0) >= self.configs.criteria_threshold:
                relevant_pages.append(score)
            else:
                discarded_pages.append(score)
        
        return relevant_pages, discarded_pages
    
    def _extract_page_numbers(self, relevant_pages: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """
        Extract page numbers from relevant pages
        
        Args:
            relevant_pages: List of relevant pages
            
        Returns:
            Dictionary mapping document IDs to lists of page numbers
        """
        page_numbers = {}
        
        for page in relevant_pages:
            doc_id = page.get("doc_id")
            page_number = page.get("page_number", 1)  # Default to page 1 if not specified
            
            if doc_id not in page_numbers:
                page_numbers[doc_id] = []
                
            page_numbers[doc_id].append(page_number)
        
        return page_numbers
    
    def _extract_cited_pages(self, docs: List[Any], 
                           page_numbers: Dict[str, List[int]]) -> List[Dict[str, Any]]:
        """
        Extract cited pages from documents
        
        Args:
            docs: List of original documents
            page_numbers: Dictionary mapping document IDs to lists of page numbers
            
        Returns:
            List of relevant page contents
        """
        # If cited page extractor service is available, use it
        if self.cited_page_extractor:
            return self.cited_page_extractor.extract(docs, page_numbers)
        
        # Fallback implementation
        cited_pages = []
        
        # Create a dict mapping document ID to original document
        doc_map = {doc.get("id"): doc for doc in docs}
        
        for doc_id, page_nums in page_numbers.items():
            doc = doc_map.get(doc_id)
            
            if not doc:
                logger.warning(f"Document not found for ID: {doc_id}")
                continue
                
            # Extract content for each page
            for page_num in page_nums:
                # In this simplified implementation, we assume the whole document is the content
                # In a real system, this would extract specific pages from multi-page documents
                cited_pages.append({
                    "doc_id": doc_id,
                    "page_number": page_num,
                    "content": doc.get("content", ""),
                    "title": doc.get("title", ""),
                    "url": doc.get("url", "")
                })
        
        return cited_pages
    
    def _create_assessment_prompt(self, doc: Dict[str, Any], 
                                variable_definition: Dict[str, Any]) -> str:
        """
        Create assessment prompt for document relevance
        
        Args:
            doc: Document to assess
            variable_definition: Variable definition and description
            
        Returns:
            Assessment prompt
        """
        # Get document content
        content = doc.get("content", "")[:5000]  # Limit content length
        
        # Get variable information
        description = variable_definition.get("description", "")
        prompt_sequence = variable_definition.get("prompt_sequence", [])
        
        # Create assessment prompt
        prompt = f"""
You are a document relevance assessor. Your task is to determine if the following document is relevant to the given information need.

Information Need: {description}

Key Questions:
{chr(10).join([f"- {p}" for p in prompt_sequence])}

Document Content:
{content}

Please assess the document's relevance to the information need based on the following criteria:
1. Does the document contain information directly related to the information need?
2. Does the document provide sufficient detail to answer at least one of the key questions?
3. Is the document from a credible source?

Provide your assessment in the following format:
RELEVANT: [Yes/No]
CONFIDENCE: [0.0-1.0]
CITATION: [Most relevant text snippet from the document that supports your assessment]
REASONING: [Brief explanation for your assessment]
"""
        return prompt
    
    def _parse_assessment(self, llm_response: str, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse LLM assessment response
        
        Args:
            llm_response: LLM response text
            doc: Original document
            
        Returns:
            Parsed assessment
        """
        assessment = {
            "doc_id": doc.get("id"),
            "relevant": False,
            "confidence": 0.0,
            "citation": "",
            "reasoning": ""
        }
        
        try:
            # Parse relevant
            if "RELEVANT: Yes" in llm_response:
                assessment["relevant"] = True
                
            # Parse confidence
            confidence_match = re.search(r"CONFIDENCE: (0\.\d+|1\.0)", llm_response)
            if confidence_match:
                assessment["confidence"] = float(confidence_match.group(1))
                
            # Parse citation
            citation_match = re.search(r"CITATION: (.*?)(?=REASONING:|$)", llm_response, re.DOTALL)
            if citation_match:
                citation = citation_match.group(1).strip()
                # Truncate if necessary
                if len(citation) > self.configs.max_citation_length:
                    citation = citation[:self.configs.max_citation_length] + "..."
                assessment["citation"] = citation
                
            # Parse reasoning
            reasoning_match = re.search(r"REASONING: (.*?)$", llm_response, re.DOTALL)
            if reasoning_match:
                assessment["reasoning"] = reasoning_match.group(1).strip()
                
        except Exception as e:
            logger.error(f"Error parsing assessment: {e}")
        
        return assessment
    
    def _create_hallucination_prompt(self, assessment: Dict[str, Any]) -> str:
        """
        Create prompt to check for hallucinations
        
        Args:
            assessment: Document assessment
            
        Returns:
            Hallucination check prompt
        """
        citation = assessment.get("citation", "")
        
        prompt = f"""
You are a fact-checking assistant. Your task is to analyze the following excerpt and determine if it directly addresses tax rates, specific tax information, or tax regulations.

Text excerpt:
{citation}

Please analyze this text and determine if it contains SPECIFIC information about tax rates, tax percentages, or tax regulations.
Answer with "HALLUCINATION: Yes" if the text does NOT contain specific tax information.
Answer with "HALLUCINATION: No" if the text DOES contain specific tax information.

Provide a brief explanation for your decision.
"""
        return prompt
    
    def _parse_hallucination_check(self, hallucination_response: str) -> bool:
        """
        Parse hallucination check response
        
        Args:
            hallucination_response: LLM response text
            
        Returns:
            True if hallucination detected, False otherwise
        """
        return "HALLUCINATION: Yes" in hallucination_response
        
import re  # Added for regex pattern matching in parsing