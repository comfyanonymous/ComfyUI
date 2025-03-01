from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class BusinessOwnerAssumptions(BaseModel):
    """Assumptions about the business owner"""
    has_annual_gross_income: str = "$70,000"
    
class BusinessAssumptions(BaseModel):
    """Assumptions about the business"""
    year_of_operation: str = "second year"
    qualifies_for_incentives: bool = False
    gross_annual_revenue: str = "$1,000,000"
    employees: int = 15
    business_type: str = "general commercial activities (NAICS: 4523)"
    
class TaxesAssumptions(BaseModel):
    """Assumptions about taxes"""
    taxes_paid_period: str = "second year of operation"
    
class OtherAssumptions(BaseModel):
    """Other assumptions"""
    other_assumptions: List[str] = Field(default_factory=list)

class Assumptions(BaseModel):
    """Collection of all assumptions"""
    business_owner: Optional[BusinessOwnerAssumptions] = None
    business: Optional[BusinessAssumptions] = None
    taxes: Optional[TaxesAssumptions] = None
    other: Optional[OtherAssumptions] = None

class PromptDecisionTreeNode(BaseModel):
    """Node in the prompt decision tree"""
    prompt: str
    depends_on: Optional[List[str]] = None
    next_prompts: Optional[Dict[str, str]] = None
    
class Variable(BaseModel):
    """Variable definition in the codebook"""
    label: str
    item_name: str
    description: str
    units: str
    assumptions: Optional[Assumptions] = None
    prompt_decision_tree: Optional[List[PromptDecisionTreeNode]] = None

class VariableCodebookConfigs(BaseModel):
    """Configuration for Variable Codebook"""
    variables_path: str = "variables.json"
    load_from_file: bool = True
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    default_assumptions_enabled: bool = True

class VariableCodebook:
    """
    Variable Codebook system based on mermaid class diagram in README.md
    Manages variable definitions and their associated assumptions and prompt sequences
    """
    
    def __init__(self, resources: Dict[str, Any], configs: VariableCodebookConfigs):
        """
        Initialize with injected dependencies and configuration
        
        Args:
            resources: Dictionary of resources including storage services
            configs: Configuration for Variable Codebook
        """
        self.resources = resources
        self.configs = configs
        
        # Extract needed services from resources
        self.storage_service = resources.get("storage_service")
        self.cache_service = resources.get("cache_service")
        
        # Initialize variables dictionary
        self.variables: Dict[str, Variable] = {}
        
        # Load variables if configured
        if self.configs.load_from_file:
            self._load_variables()
            
        logger.info("VariableCodebook initialized with services")

    def control_flow(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Execute variable codebook operations based on the action
        
        Args:
            action: Operation to perform (get_variable, get_prompt_sequence, etc.)
            **kwargs: Operation-specific parameters
            
        Returns:
            Dictionary containing operation results
        """
        logger.info(f"Starting variable codebook operation: {action}")
        
        if action == "get_variable":
            return self._get_variable(
                variable_name=kwargs.get("variable_name", "")
            )
        elif action == "get_prompt_sequence":
            return self._get_prompt_sequence(
                variable_name=kwargs.get("variable_name", ""),
                input_data_point=kwargs.get("input_data_point", "")
            )
        elif action == "get_assumptions":
            return self._get_assumptions(
                variable_name=kwargs.get("variable_name", "")
            )
        elif action == "add_variable":
            return self._add_variable(
                variable=kwargs.get("variable")
            )
        elif action == "update_variable":
            return self._update_variable(
                variable_name=kwargs.get("variable_name", ""),
                variable=kwargs.get("variable")
            )
        else:
            logger.error(f"Unknown action: {action}")
            raise ValueError(f"Unknown action: {action}")

    def get_prompt_sequence_for_input(self, input_data_point: str) -> List[str]:
        """
        Get prompt sequence for a given input data point
        
        Args:
            input_data_point: The query or information request
            
        Returns:
            List of prompts in the sequence
        """
        # Extract variable name from input data point
        variable_name = self._extract_variable_from_input(input_data_point)
        
        # Get prompt sequence for the variable
        result = self.control_flow(
            "get_prompt_sequence",
            variable_name=variable_name,
            input_data_point=input_data_point
        )
        
        return result.get("prompt_sequence", [])

    def _extract_variable_from_input(self, input_data_point: str) -> str:
        """
        Extract the variable name from the input data point
        
        This is a simplified implementation that uses keyword matching.
        In a real system, this could use NLP techniques or more sophisticated parsing.
        
        Args:
            input_data_point: The query or information request
            
        Returns:
            Variable name
        """
        # Convert to lowercase for case-insensitive matching
        input_lower = input_data_point.lower()
        
        # Define keyword mappings to variable names
        keyword_mappings = {
            "sales tax": "sales_tax_city",
            "tax rate": "sales_tax_city",
            "local tax": "sales_tax_city",
            "city tax": "sales_tax_city",
            "municipal tax": "sales_tax_city",
            "property tax": "property_tax",
            "income tax": "income_tax"
        }
        
        # Find the first matching keyword
        for keyword, variable in keyword_mappings.items():
            if keyword in input_lower:
                return variable
        
        # Default to a generic variable if no match is found
        return "generic_tax_information"

    def _get_variable(self, variable_name: str) -> Dict[str, Any]:
        """
        Get a variable from the codebook
        
        Args:
            variable_name: Name of the variable
            
        Returns:
            Dictionary containing the variable information
        """
        if variable_name in self.variables:
            return {
                "success": True,
                "variable": self.variables[variable_name]
            }
        else:
            logger.warning(f"Variable not found: {variable_name}")
            return {
                "success": False,
                "error": f"Variable not found: {variable_name}"
            }

    def _get_prompt_sequence(self, variable_name: str, input_data_point: str) -> Dict[str, Any]:
        """
        Get the prompt sequence for a variable
        
        Args:
            variable_name: Name of the variable
            input_data_point: The query or information request
            
        Returns:
            Dictionary containing the prompt sequence
        """
        # Get the variable
        variable_result = self._get_variable(variable_name)
        
        if not variable_result.get("success", False):
            return variable_result
            
        variable = variable_result.get("variable")
        
        # Extract the prompt sequence from the variable
        if not variable.prompt_decision_tree:
            logger.warning(f"No prompt decision tree found for variable: {variable_name}")
            return {
                "success": False,
                "error": f"No prompt decision tree found for variable: {variable_name}"
            }
            
        # Extract prompts from the decision tree
        prompts = [node.prompt for node in variable.prompt_decision_tree]
        
        return {
            "success": True,
            "prompt_sequence": prompts,
            "variable": variable
        }

    def _get_assumptions(self, variable_name: str) -> Dict[str, Any]:
        """
        Get the assumptions for a variable
        
        Args:
            variable_name: Name of the variable
            
        Returns:
            Dictionary containing the assumptions
        """
        # Get the variable
        variable_result = self._get_variable(variable_name)
        
        if not variable_result.get("success", False):
            return variable_result
            
        variable = variable_result.get("variable")
        
        # Extract the assumptions from the variable
        return {
            "success": True,
            "assumptions": variable.assumptions,
            "variable": variable
        }

    def _add_variable(self, variable: Variable) -> Dict[str, Any]:
        """
        Add a variable to the codebook
        
        Args:
            variable: Variable to add
            
        Returns:
            Dictionary containing the operation result
        """
        if variable.item_name in self.variables:
            logger.warning(f"Variable already exists: {variable.item_name}")
            return {
                "success": False,
                "error": f"Variable already exists: {variable.item_name}"
            }
            
        # Add the variable
        self.variables[variable.item_name] = variable
        
        # Save to storage if available
        if self.storage_service:
            self.storage_service.save_variable(variable)
        
        return {
            "success": True,
            "variable": variable
        }

    def _update_variable(self, variable_name: str, variable: Variable) -> Dict[str, Any]:
        """
        Update a variable in the codebook
        
        Args:
            variable_name: Name of the variable to update
            variable: Updated variable
            
        Returns:
            Dictionary containing the operation result
        """
        if variable_name not in self.variables:
            logger.warning(f"Variable not found: {variable_name}")
            return {
                "success": False,
                "error": f"Variable not found: {variable_name}"
            }
            
        # Update the variable
        self.variables[variable_name] = variable
        
        # Save to storage if available
        if self.storage_service:
            self.storage_service.save_variable(variable)
        
        return {
            "success": True,
            "variable": variable
        }

    def _load_variables(self) -> None:
        """Load variables from storage"""
        try:
            if self.storage_service:
                variables = self.storage_service.load_variables(self.configs.variables_path)
                
                if variables:
                    self.variables = {var.item_name: var for var in variables}
                    logger.info(f"Loaded {len(self.variables)} variables from storage")
                else:
                    logger.warning("No variables found in storage")
                    self._load_default_variables()
            else:
                logger.warning("No storage service available, loading default variables")
                self._load_default_variables()
        except Exception as e:
            logger.error(f"Error loading variables: {e}")
            self._load_default_variables()
            
    def _load_default_variables(self) -> None:
        """Load default variables"""
        if not self.configs.default_assumptions_enabled:
            logger.info("Default assumptions disabled, skipping default variable loading")
            return
            
        # Create a sample variable with assumptions and prompt decision tree
        sales_tax_variable = Variable(
            label="Sales Tax - City",
            item_name="sales_tax_city",
            description="A tax levied on the sales of all goods and services by the municipal government.",
            units="Double (Percent)",
            assumptions=Assumptions(
                business_owner=BusinessOwnerAssumptions(),
                business=BusinessAssumptions(),
                taxes=TaxesAssumptions(),
                other=OtherAssumptions(
                    other_assumptions=["Also assume the business has no special tax exemptions."]
                )
            ),
            prompt_decision_tree=[
                PromptDecisionTreeNode(
                    prompt="List the name of the tax as given in the document verbatim, as well as its line item."
                ),
                PromptDecisionTreeNode(
                    prompt="List the formal definition of the tax verbatim, as well as its line item."
                ),
                PromptDecisionTreeNode(
                    prompt="Does this statute apply to all goods or services, or only to specific ones?"
                ),
                PromptDecisionTreeNode(
                    prompt="What is the exact percentage rate of the tax?"
                )
            ]
        )
        
        # Add to variables dictionary
        self.variables[sales_tax_variable.item_name] = sales_tax_variable
        
        logger.info("Loaded default variables")