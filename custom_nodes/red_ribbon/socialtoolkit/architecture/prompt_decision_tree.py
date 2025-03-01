from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Union
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class PromptDecisionTreeConfigs(BaseModel):
    """Configuration for Prompt Decision Tree workflow"""
    max_tokens_per_prompt: int = 2000
    max_pages_to_concatenate: int = 10
    max_iterations: int = 5
    confidence_threshold: float = 0.7
    enable_human_review: bool = True  # Whether to enable human review for low confidence or errors
    context_window_size: int = 8000  # Maximum context window size for LLM

class PromptDecisionTreeNodeType(str, Enum):
    """Types of nodes in the prompt decision tree"""
    QUESTION = "question"
    DECISION = "decision"
    FINAL = "final"

class PromptDecisionTreeEdge(BaseModel):
    """Edge in the prompt decision tree"""
    condition: str
    next_node_id: str

class PromptDecisionTreeNode(BaseModel):
    """Node in the prompt decision tree"""
    id: str
    type: PromptDecisionTreeNodeType
    prompt: str
    edges: Optional[List[PromptDecisionTreeEdge]] = None
    is_final: bool = False

class PromptDecisionTree:
    """
    Prompt Decision Tree system based on mermaid flowchart in README.md
    Executes a decision tree of prompts to extract information from documents
    """
    
    def __init__(self, resources: Dict[str, Any], configs: PromptDecisionTreeConfigs):
        """
        Initialize with injected dependencies and configuration
        
        Args:
            resources: Dictionary of resources including services
            configs: Configuration for Prompt Decision Tree
        """
        self.resources = resources
        self.configs = configs
        
        # Extract needed services from resources
        self.variable_codebook = resources.get("variable_codebook_service")
        self.human_review_service = resources.get("human_review_service")
        
        logger.info("PromptDecisionTree initialized with services")

    def control_flow(self, relevant_pages: List[Any], 
                   prompt_sequence: List[str],
                   llm_api: Any) -> Dict[str, Any]:
        """
        Execute the prompt decision tree flow based on the mermaid flowchart
        
        Args:
            relevant_pages: List of relevant document pages
            prompt_sequence: List of prompts in the decision tree
            llm_api: LLM API instance
            
        Returns:
            Dictionary containing the output data point
        """
        logger.info(f"Starting prompt decision tree with {len(relevant_pages)} pages")
        
        # Step 1: Concatenate pages
        concatenated_pages = self._concatenate_pages(relevant_pages)
        
        # Step 2: Get desired data point codebook entry & prompt sequence
        # (Already provided as input parameter)
        
        # Step 3: Execute prompt decision tree
        result = self._execute_decision_tree(
            concatenated_pages, prompt_sequence, llm_api
        )
        
        # Step 4: Handle errors and unforeseen edgecases if needed
        if result.get("error") and self.configs.enable_human_review:
            result = self._request_human_review(result, concatenated_pages)
        
        logger.info("Completed prompt decision tree execution")
        return result

    def execute(self, relevant_pages: List[Any], prompt_sequence: List[str], 
               llm_api: Any) -> Any:
        """
        Public method to execute prompt decision tree
        
        Args:
            relevant_pages: List of relevant document pages
            prompt_sequence: List of prompts in the decision tree
            llm_api: LLM API instance
            
        Returns:
            Output data point
        """
        result = self.control_flow(relevant_pages, prompt_sequence, llm_api)
        return result.get("output_data_point", "")
    
    def _concatenate_pages(self, pages: List[Any]) -> str:
        """
        Concatenate pages into a single document
        
        Args:
            pages: List of pages to concatenate
            
        Returns:
            Concatenated document text
        """
        # Limit number of pages to avoid context window issues
        pages_to_use = pages[:self.configs.max_pages_to_concatenate]
        
        concatenated_text = ""
        
        for i, page in enumerate(pages_to_use):
            content = page.get("content", "")
            title = page.get("title", f"Document {i+1}")
            url = page.get("url", "")
            
            page_text = f"""
--- DOCUMENT {i+1}: {title} ---
Source: {url}

{content}

"""
            concatenated_text += page_text
            
        return concatenated_text
    
    def _execute_decision_tree(self, document_text: str, 
                             prompt_sequence: List[str], 
                             llm_api: Any) -> Dict[str, Any]:
        """
        Execute the prompt decision tree
        
        Args:
            document_text: Concatenated document text
            prompt_sequence: List of prompts in the decision tree
            llm_api: LLM API instance
            
        Returns:
            Dictionary containing the execution result
        """
        # Create a simplified decision tree from the prompt sequence
        decision_tree = self._create_decision_tree(prompt_sequence)
        
        try:
            # Start with the first node
            current_node = decision_tree[0]
            iteration = 0
            responses = []
            
            # Follow the decision tree until a final node is reached or max iterations is exceeded
            while not current_node.is_final and iteration < self.configs.max_iterations:
                # Generate prompt for the current node
                prompt = self._generate_node_prompt(current_node, document_text)
                
                # Get response from LLM
                llm_response = llm_api.generate(prompt, max_tokens=self.configs.max_tokens_per_prompt)
                responses.append({
                    "node_id": current_node.id,
                    "prompt": prompt,
                    "response": llm_response
                })
                
                # Determine next node based on response
                if current_node.edges:
                    next_node_id = self._determine_next_node(llm_response, current_node.edges)
                    current_node = next(
                        (node for node in decision_tree if node.id == next_node_id), 
                        decision_tree[-1]  # Default to the last node if not found
                    )
                else:
                    # No edges, move to the next node in sequence
                    node_index = decision_tree.index(current_node)
                    if node_index + 1 < len(decision_tree):
                        current_node = decision_tree[node_index + 1]
                    else:
                        # End of sequence, mark as final
                        current_node.is_final = True
                
                iteration += 1
            
            # Process the final response
            final_response = responses[-1]["response"] if responses else ""
            output_data_point = self._extract_output_data_point(final_response)
            
            return {
                "success": True,
                "output_data_point": output_data_point,
                "responses": responses,
                "iterations": iteration
            }
            
        except Exception as e:
            logger.error(f"Error executing decision tree: {e}")
            return {
                "success": False,
                "error": str(e),
                "output_data_point": ""
            }
    
    def _create_decision_tree(self, prompt_sequence: List[str]) -> List[PromptDecisionTreeNode]:
        """
        Create a decision tree from a prompt sequence
        
        This is a simplified implementation that creates a linear sequence of nodes.
        In a real system, this would create a proper tree structure with branches.
        
        Args:
            prompt_sequence: List of prompts
            
        Returns:
            List of nodes in the decision tree
        """
        nodes = []
        
        for i, prompt in enumerate(prompt_sequence):
            # Create a node for each prompt
            node = PromptDecisionTreeNode(
                id=f"node_{i}",
                type=PromptDecisionTreeNodeType.QUESTION,
                prompt=prompt,
                is_final=(i == len(prompt_sequence) - 1)  # Last node is final
            )
            
            # Add edges if not the last node
            if i < len(prompt_sequence) - 1:
                node.edges = [
                    PromptDecisionTreeEdge(
                        condition="default",
                        next_node_id=f"node_{i+1}"
                    )
                ]
                
            nodes.append(node)
        
        return nodes
    
    def _generate_node_prompt(self, node: PromptDecisionTreeNode, document_text: str) -> str:
        """
        Generate a prompt for a node in the decision tree
        
        Args:
            node: Current node in the decision tree
            document_text: Document text
            
        Returns:
            Prompt for the node
        """
        # Truncate document text if too long
        max_doc_length = self.configs.context_window_size - 500  # Reserve space for instructions
        if len(document_text) > max_doc_length:
            document_text = document_text[:max_doc_length] + "..."
            
        prompt = f"""
You are an expert tax researcher assisting with data extraction from official documents.
Please carefully analyze the following documents to answer this specific question:

QUESTION: {node.prompt}

DOCUMENTS:
{document_text}

Based solely on the information provided in these documents, please answer the question above.
If the answer is explicitly stated in the documents, provide the exact information along with its source.
If the answer requires interpretation, explain your reasoning clearly.
If the information is not available in the documents, respond with "Information not available in the provided documents."

Your answer should be concise, factual, and directly address the question.
"""
        return prompt
        
    def _determine_next_node(self, response: str, edges: List[PromptDecisionTreeEdge]) -> str:
        """
        Determine the next node based on the response
        
        This is a simplified implementation that just follows the default edge.
        In a real system, this would analyze the response to determine the path.
        
        Args:
            response: LLM response
            edges: List of edges from the current node
            
        Returns:
            ID of the next node
        """
        # In this simplified version, just follow the first edge
        if edges:
            return edges[0].next_node_id
        return ""
    
    def _extract_output_data_point(self, response: str) -> str:
        """
        Extract the output data point from the final response
        
        Args:
            response: Final LLM response
            
        Returns:
            Extracted output data point
        """
        # Look for patterns like "X%" or "X percent"
        import re
        
        # Try to find percentage patterns
        percentage_match = re.search(r'(\d+(?:\.\d+)?)\s*%', response)
        if percentage_match:
            return percentage_match.group(0)
            
        percentage_word_match = re.search(r'(\d+(?:\.\d+)?)\s+percent', response, re.IGNORECASE)
        if percentage_word_match:
            value = percentage_word_match.group(1)
            return f"{value}%"
            
        # Look for specific statements about rates
        rate_match = re.search(r'rate\s+is\s+(\d+(?:\.\d+)?)', response, re.IGNORECASE)
        if rate_match:
            value = rate_match.group(1)
            return f"{value}%"
            
        # If no specific patterns are found, return a cleaned up version of the response
        # Limit to 100 characters for brevity
        cleaned_response = response.strip()
        if len(cleaned_response) > 100:
            cleaned_response = cleaned_response[:97] + "..."
            
        return cleaned_response
    
    def _request_human_review(self, result: Dict[str, Any], document_text: str) -> Dict[str, Any]:
        """
        Request human review for errors or low confidence results
        
        Args:
            result: Result from decision tree execution
            document_text: Document text
            
        Returns:
            Updated result after human review
        """
        if self.human_review_service:
            review_request = {
                "error": result.get("error"),
                "document_text": document_text,
                "responses": result.get("responses", [])
            }
            
            human_review_result = self.human_review_service.review(review_request)
            
            if human_review_result.get("success"):
                result["output_data_point"] = human_review_result.get("output_data_point", "")
                result["human_reviewed"] = True
                result["success"] = True
                result.pop("error", None)
        
        return result