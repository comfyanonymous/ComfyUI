import pytest
import json
from unittest.mock import patch, MagicMock
import sys
import os

# Import the execution module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from execution import execute_workflow


@pytest.mark.parametrize("auth_header,expected_status", [
    (None, 401),  # Missing authentication token
    ("Bearer expired_token_xyz", 401),  # Expired/invalid token
    ("Bearer malformed", 401),  # Malformed token format
    ("", 401),  # Empty auth header
    ("InvalidScheme token123", 401),  # Invalid auth scheme
])
def test_unauthenticated_workflow_execution_rejected(auth_header, expected_status):
    """Invariant: Protected endpoints reject unauthenticated requests with 401/403 status"""
    
    # Mock workflow payload that would execute arbitrary code
    malicious_workflow = {
        "nodes": {
            "1": {
                "class_type": "CheckpointLoader",
                "inputs": {"ckpt_name": "model.safetensors"}
            }
        }
    }
    
    # Mock the request context to simulate unauthenticated API call
    with patch('execution.request') as mock_request:
        mock_request.headers = {"Authorization": auth_header} if auth_header else {}
        
        # Mock authentication check that should reject unauthenticated requests
        with patch('execution.validate_auth') as mock_auth:
            mock_auth.return_value = False
            
            # Attempt to execute workflow without valid credentials
            result = execute_workflow(malicious_workflow, auth_header)
            
            # Assert that execution is rejected
            assert result.get("status") == "error" or result.get("code") in [401, 403], \
                f"Unauthenticated request should be rejected, got: {result}"
            mock_auth.assert_called()