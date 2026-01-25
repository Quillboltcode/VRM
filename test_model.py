#!/usr/bin/env python3
"""
Test script to verify the model architecture works correctly.
"""

import torch
from model import RecursiveFERModel

def test_model():
    """Test the model with dummy input."""
    print("Testing RecursiveFERModel model...")
    
    # Model parameters
    batch_size = 4
    in_channels = 3
    num_classes = 7
    hidden_dim = 64
    max_steps = 5
    image_size = 224
    
    # Initialize model
    model = RecursiveFERModel(
        in_channels=in_channels,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        max_steps=max_steps
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, in_channels, image_size, image_size)
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    try:
        model.eval()
        with torch.no_grad():
            print("Debug: About to call model forward...")
            logits, num_steps = model(dummy_input)
            print("Debug: Model forward completed")
        
        print(f"Output logits shape: {logits.shape}")
        print(f"Number of steps per sample: {num_steps}")
        print(f"Average steps: {num_steps.float().mean():.2f}")
        
        # Verify output shapes
        assert logits.shape == (batch_size, num_classes), f"Expected logits shape {(batch_size, num_classes)}, got {logits.shape}"
        assert num_steps.shape == (batch_size,), f"Expected num_steps shape {(batch_size,)}, got {num_steps.shape}"
        assert torch.all(num_steps >= 1) and torch.all(num_steps <= max_steps), "Steps should be between 1 and max_steps"
        
        print("✅ Model test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_loss_integration():
    """Test loss function integration with model."""
    print("\nTesting loss function integration...")
    
    from loss import PonderLoss
    
    # Model and loss parameters
    batch_size = 4
    num_classes = 7
    hidden_dim = 64
    max_steps = 5
    
    # Initialize model and loss
    model = RecursiveFERModel(
        in_channels=3,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        max_steps=max_steps
    )
    
    classification_loss = torch.nn.CrossEntropyLoss()
    criterion = PonderLoss(classification_loss, lambda_ponder=0.01)
    
    # Create dummy data
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    dummy_labels = torch.randint(0, num_classes, (batch_size,))
    
    try:
        with torch.no_grad():
            logits, num_steps = model(dummy_input)
        
        # Calculate loss
        total_loss, class_loss, ponder_cost = criterion(logits, dummy_labels, num_steps)
        
        print(f"Total loss: {total_loss.item():.4f}")
        print(f"Classification loss: {class_loss.item():.4f}")
        print(f"Ponder cost: {ponder_cost.item():.4f}")
        
        # Verify loss values are reasonable
        assert total_loss.item() >= 0, "Total loss should be non-negative"
        assert class_loss.item() >= 0, "Classification loss should be non-negative"
        assert ponder_cost.item() >= 0, "Ponder cost should be non-negative"
        
        print("✅ Loss integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Loss integration test failed: {e}")
        return False

if __name__ == "__main__":
    model_test_passed = test_model()
    loss_test_passed = test_loss_integration()
    
    if model_test_passed and loss_test_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Some tests failed!")