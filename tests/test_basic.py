#!/usr/bin/env python
"""
Basic tests for EEG Super-Resolution components

This file contains basic unit tests to verify that the main components
of the EEG super-resolution system are working correctly.
"""

import unittest
import sys
import os
import torch
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from modeling_pretrain import reconstruction_base_patch250_250
from utils import get_model

class TestEEGSuperResolution(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 2
        self.n_channels = 9
        self.n_timepoints = 250
        
    def test_model_creation(self):
        """Test that the model can be created successfully."""
        try:
            model = reconstruction_base_patch250_250(pretrained=False)
            self.assertIsNotNone(model)
            print("✓ Model creation test passed")
        except Exception as e:
            self.fail(f"Model creation failed: {e}")
    
    def test_model_forward_pass(self):
        """Test that the model can perform a forward pass."""
        try:
            model = reconstruction_base_patch250_250(pretrained=False)
            model = model.to(self.device)
            model.eval()
            
            # Create sample input
            x = torch.randn(self.batch_size, self.n_channels, self.n_timepoints)
            x = x.to(self.device)
            
            with torch.no_grad():
                output = model(x)
            
            # Check output shape
            self.assertEqual(output.shape[0], self.batch_size)
            self.assertEqual(output.shape[2], self.n_timepoints)
            
            print(f"✓ Forward pass test passed. Input: {x.shape}, Output: {output.shape}")
        except Exception as e:
            self.fail(f"Forward pass failed: {e}")
    
    def test_model_parameters(self):
        """Test that the model has reasonable number of parameters."""
        try:
            model = reconstruction_base_patch250_250(pretrained=False)
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Check that we have a reasonable number of parameters
            self.assertGreater(total_params, 1000)  # At least 1K parameters
            self.assertLess(total_params, 1e9)      # Less than 1B parameters
            self.assertEqual(total_params, trainable_params)  # All parameters trainable by default
            
            print(f"✓ Parameter test passed. Total: {total_params:,}, Trainable: {trainable_params:,}")
        except Exception as e:
            self.fail(f"Parameter test failed: {e}")
    
    def test_model_device_compatibility(self):
        """Test that the model works on both CPU and GPU (if available)."""
        try:
            model = reconstruction_base_patch250_250(pretrained=False)
            
            # Test CPU
            model_cpu = model.to('cpu')
            x_cpu = torch.randn(1, self.n_channels, self.n_timepoints)
            
            with torch.no_grad():
                output_cpu = model_cpu(x_cpu)
            
            self.assertEqual(output_cpu.device.type, 'cpu')
            
            # Test GPU if available
            if torch.cuda.is_available():
                model_gpu = model.to('cuda')
                x_gpu = torch.randn(1, self.n_channels, self.n_timepoints).to('cuda')
                
                with torch.no_grad():
                    output_gpu = model_gpu(x_gpu)
                
                self.assertEqual(output_gpu.device.type, 'cuda')
                print("✓ Device compatibility test passed (CPU + GPU)")
            else:
                print("✓ Device compatibility test passed (CPU only)")
                
        except Exception as e:
            self.fail(f"Device compatibility test failed: {e}")
    
    def test_model_deterministic(self):
        """Test that the model produces deterministic outputs with same input."""
        try:
            # Set seed for reproducibility
            torch.manual_seed(42)
            np.random.seed(42)
            
            model = reconstruction_base_patch250_250(pretrained=False)
            model = model.to(self.device)
            model.eval()
            
            x = torch.randn(1, self.n_channels, self.n_timepoints).to(self.device)
            
            with torch.no_grad():
                output1 = model(x)
                output2 = model(x)
            
            # Check outputs are identical
            self.assertTrue(torch.allclose(output1, output2, atol=1e-6))
            print("✓ Deterministic output test passed")
            
        except Exception as e:
            self.fail(f"Deterministic test failed: {e}")

class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_imports(self):
        """Test that all required modules can be imported."""
        try:
            import torch
            import numpy as np
            import matplotlib
            import scipy
            import sklearn
            print("✓ Import test passed")
        except ImportError as e:
            self.fail(f"Import test failed: {e}")

def run_tests():
    """Run all tests and provide a summary."""
    print("Running EEG Super-Resolution Tests")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestEEGSuperResolution))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("🎉 All tests passed!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1) 