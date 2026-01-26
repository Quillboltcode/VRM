#!/usr/bin/env python3
"""
Test script to verify RecursiveFER model architecture works correctly.
Tests both inference and training modes with supervised halting.
"""

import torch
import torch.nn.functional as F
from model import RecursiveFER

def test_inference():
    """Test model inference mode (without labels)."""
    print("Testing RecursiveFER inference mode...")
    
    # Model parameters
    batch_size = 4
    in_channels = 3
    num_classes = 7
    hidden_dim = 64
    max_steps = 5
    image_size = 224
    
    # Initialize model
    model = RecursiveFER(
        in_channels=in_channels,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        max_steps=max_steps
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, in_channels, image_size, image_size)
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass (inference)
    try:
        model.eval()
        with torch.no_grad():
            logits, steps_taken, halt_probs = model(dummy_input)
        
        print(f"Output logits shape: {logits.shape}")
        print(f"Steps taken per sample: {steps_taken}")
        print(f"Halt probabilities shape: {halt_probs.shape}")
        print(f"Average steps: {steps_taken.float().mean():.2f}")
        print(f"Halt probabilities range: [{halt_probs.min():.3f}, {halt_probs.max():.3f}]")
        
        # Verify output shapes and values
        assert logits.shape == (batch_size, num_classes), f"Expected logits shape {(batch_size, num_classes)}, got {logits.shape}"
        assert steps_taken.shape == (batch_size,), f"Expected steps_taken shape {(batch_size,)}, got {steps_taken.shape}"
        assert halt_probs.shape == (batch_size, max_steps), f"Expected halt_probs shape {(batch_size, max_steps)}, got {halt_probs.shape}"
        assert torch.all(steps_taken >= 1) and torch.all(steps_taken <= max_steps), "Steps should be between 1 and max_steps"
        assert torch.all(halt_probs >= 0) and torch.all(halt_probs <= 1), "Halt probabilities should be between 0 and 1"
        
        # Test softmax probabilities
        softmax_probs = F.softmax(logits, dim=1)
        assert torch.allclose(softmax_probs.sum(dim=1), torch.ones(batch_size)), "Softmax probabilities should sum to 1"
        
        print("✅ Inference test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training():
    """Test model training mode (with labels)."""
    print("\nTesting RecursiveFER training mode...")
    
    # Model parameters
    batch_size = 4
    num_classes = 7
    hidden_dim = 64
    max_steps = 5
    
    # Initialize model
    model = RecursiveFER(
        in_channels=3,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        max_steps=max_steps
    )
    
    # Create dummy data
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    dummy_labels = torch.randint(0, num_classes, (batch_size,))
    print(f"Input shape: {dummy_input.shape}")
    print(f"Labels: {dummy_labels}")
    
    try:
        model.train()
        total_loss, (task_loss, halt_loss) = model(dummy_input, dummy_labels)
        
        print(f"Total loss: {total_loss.item():.4f}")
        print(f"Task loss (cross-entropy): {task_loss.item():.4f}")
        print(f"Halt loss (binary cross-entropy): {halt_loss.item():.4f}")
        
        # Verify loss values are reasonable
        assert total_loss.item() >= 0, "Total loss should be non-negative"
        assert task_loss.item() >= 0, "Task loss should be non-negative"
        assert halt_loss.item() >= 0, "Halt loss should be non-negative"
        assert isinstance(total_loss, torch.Tensor) and total_loss.requires_grad, "Total loss should be a tensor with gradients"
        
        print("✅ Training test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradient_flow():
    """Test that gradients flow correctly through the model."""
    print("\nTesting gradient flow...")
    
    model = RecursiveFER(
        in_channels=3,
        num_classes=7,
        hidden_dim=64,
        max_steps=5
    )
    
    # Create dummy data
    dummy_input = torch.randn(2, 3, 224, 224, requires_grad=True)
    dummy_labels = torch.randint(0, 7, (2,))
    
    try:
        model.train()
        total_loss, (task_loss, halt_loss) = model(dummy_input, dummy_labels)
        total_loss.backward()
        
        # Check gradients exist
        assert dummy_input.grad is not None, "Input gradients should exist"
        has_param_grads = any(p.grad is not None for p in model.parameters())
        assert has_param_grads, "Some model parameters should have gradients"
        
        print(f"Input gradient norm: {dummy_input.grad.norm():.4f}")
        param_grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for p in model.parameters())
        print(f"Parameters with gradients: {param_grad_count}/{total_params}")
        
        print("✅ Gradient flow test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_consistency():
    """Test that model outputs are consistent across runs."""
    print("\nTesting model consistency...")
    
    model = RecursiveFER(
        in_channels=3,
        num_classes=7,
        hidden_dim=64,
        max_steps=5
    )
    
    # Same input, multiple runs
    dummy_input = torch.randn(2, 3, 224, 224)
    
    try:
        model.eval()
        with torch.no_grad():
            outputs1 = model(dummy_input)
            outputs2 = model(dummy_input)
        
        # Check consistency
        logits_match = torch.allclose(outputs1[0], outputs2[0])
        steps_match = torch.allclose(outputs1[1], outputs2[1])
        probs_match = torch.allclose(outputs1[2], outputs2[2])
        
        print(f"Logits consistent: {logits_match}")
        print(f"Steps consistent: {steps_match}")
        print(f"Halt probs consistent: {probs_match}")
        
        assert logits_match and steps_match and probs_match, "Model outputs should be consistent in eval mode"
        
        print("✅ Consistency test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_configurations():
    """Test model with different configurations."""
    print("\nTesting different model configurations...")
    
    configs = [
        {"hidden_dim": 32, "max_steps": 3, "num_classes": 5},
        {"hidden_dim": 128, "max_steps": 10, "num_classes": 10},
        {"hidden_dim": 64, "max_steps": 7, "num_classes": 3},
    ]
    
    for i, config in enumerate(configs):
        print(f"  Testing config {i+1}: {config}")
        try:
            model = RecursiveFER(
                in_channels=3,
                **config
            )
            
            dummy_input = torch.randn(2, 3, 224, 224)
            dummy_labels = torch.randint(0, config["num_classes"], (2,))
            
            # Test both modes
            model.eval()
            with torch.no_grad():
                logits, steps_taken, halt_probs = model(dummy_input)
            
            model.train()
            total_loss, (task_loss, halt_loss) = model(dummy_input, dummy_labels)
            
            # Basic checks
            batch_size = 2
            assert logits.shape == (batch_size, config["num_classes"])
            assert steps_taken.shape == (batch_size,)
            assert halt_probs.shape == (batch_size, config["max_steps"])
            assert torch.all(steps_taken >= 1) and torch.all(steps_taken <= config["max_steps"])
            
            print(f"    ✅ Config {i+1} passed")
            
        except Exception as e:
            print(f"    ❌ Config {i+1} failed: {e}")
            return False
    
    print("✅ Configuration test passed!")
    return True

if __name__ == "__main__":
    print("🧪 Running RecursiveFER Model Tests")
    print("="*50)
    
    # Run all tests
    tests = [
        ("Inference", test_inference),
        ("Training", test_training),
        ("Gradient Flow", test_gradient_flow),
        ("Consistency", test_model_consistency),
        ("Configurations", test_different_configurations),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print(f"\n{'='*50}")
    print("🏁 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Model is working correctly.")
    else:
        print("💥 Some tests failed. Check the output above for details.")