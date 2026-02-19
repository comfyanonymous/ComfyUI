"""
Unit tests for sampler scheduler functions.

Tests that schedulers consistently return the correct number of sigma values.
Related to issue #12485 and the fix for ddim_scheduler.
"""
import pytest
import torch
from comfy import samplers


class MockModelSampling:
    """Mock model sampling object for testing schedulers."""
    
    def __init__(self, num_sigmas=1000, sigma_min=0.0292, sigma_max=14.6146):
        self.sigmas = torch.linspace(sigma_max, sigma_min, num_sigmas)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def timestep(self, sigma):
        if isinstance(sigma, torch.Tensor):
            return sigma
        return torch.tensor(sigma, dtype=torch.float32)
    
    def sigma(self, timestep):
        if isinstance(timestep, torch.Tensor):
            return timestep
        return torch.tensor(timestep, dtype=torch.float32)


class TestDDIMSchedulerFix:
    """Test the fix for ddim_scheduler sigma count bug (issue #12485)."""
    
    @pytest.mark.parametrize("steps", [36, 37, 38, 39, 40])
    def test_issue_12485_regression(self, steps):
        """
        Regression test for issue #12485.
        
        The bug: ddim_scheduler returned varying numbers of sigmas
        depending on the model's sigma schedule length.
        
        The fix: Explicitly limit collection to exactly 'steps' iterations.
        """
        model_sampling = MockModelSampling(num_sigmas=1000)
        sigmas = samplers.ddim_scheduler(model_sampling, steps)
        
        expected = steps + 1
        assert len(sigmas) == expected, \
            f"Expected {expected} sigmas for {steps} steps, got {len(sigmas)}"
    
    def test_ddim_ends_with_zero(self):
        """Verify final sigma is zero."""
        model_sampling = MockModelSampling()
        sigmas = samplers.ddim_scheduler(model_sampling, 20)
        assert sigmas[-1] < 0.001
    
    def test_main_schedulers_consistent(self):
        """Verify main schedulers return steps+1 sigmas.
        
        Note: Some schedulers like 'beta' may return fewer due to
        duplicate removal, so we only test the main ones here.
        """
        model_sampling = MockModelSampling()
        steps = 38
        
        for scheduler in ["simple", "normal", "ddim_uniform"]:
            sigmas = samplers.calculate_sigmas(model_sampling, scheduler, steps)
            assert len(sigmas) == steps + 1, \
                f"Scheduler '{scheduler}' failed: got {len(sigmas)} sigmas"
