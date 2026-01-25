#!/usr/bin/env python3
"""
Test script to verify the analysis.py implementation works correctly.
"""

import os
import torch
import pandas as pd
from model import RecursiveFERModel
from dataset import RafDBDataset, get_default_transform
from torch.utils.data import DataLoader
from analysis import ModelAnalyzer

def create_dummy_checkpoint():
    """Create a dummy model checkpoint for testing."""
    model = RecursiveFERModel(in_channels=3, num_classes=7, hidden_dim=128, max_steps=10)
    checkpoint_path = "dummy_model.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Created dummy checkpoint: {checkpoint_path}")
    return checkpoint_path

def create_dummy_dataset():
    """Create dummy dataset for testing if real one doesn't exist."""
    dummy_data_dir = "./dummy_data"
    if not os.path.exists(f"{dummy_data_dir}/EmoLabel"):
        print("Creating dummy dataset structure...")
        os.makedirs(f"{dummy_data_dir}/EmoLabel", exist_ok=True)
        os.makedirs(f"{dummy_data_dir}/Image/aligned", exist_ok=True)
        
        # Create dummy label file
        with open(f"{dummy_data_dir}/EmoLabel/list_patition_label.txt", "w") as f:
            for i in range(20):
                f.write(f"test_{i+1:04d}_aligned.jpg {((i % 7) + 1)}\n")
        
        # Create dummy images
        from PIL import Image
        for i in range(20):
            img = Image.new('RGB', (100, 100), color=(i*12, i*8, i*5))
            img.save(f"{dummy_data_dir}/Image/aligned/test_{i+1:04d}_aligned.jpg")
    
    return dummy_data_dir

def test_analyzer():
    """Test the ModelAnalyzer class."""
    print("Testing ModelAnalyzer...")
    
    # Create dummy checkpoint and dataset
    checkpoint_path = create_dummy_checkpoint()
    create_dummy_dataset()
    
    # Initialize analyzer
    analyzer = ModelAnalyzer(checkpoint_path, device="cpu")
    
    # Test model loading
    try:
        model = analyzer.load_model()
        print("✅ Model loading test passed!")
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False
    
    # Test data loading
    try:
        dummy_data_dir = create_dummy_dataset()
        test_transform = get_default_transform(image_size=224, is_train=False)
        label_file = f"{dummy_data_dir}/EmoLabel/list_patition_label.txt"
        test_dataset = RafDBDataset(dummy_data_dir, label_file, transform=test_transform, is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
        
        print(f"✅ Dataset loading test passed! Found {len(test_dataset)} samples.")
    except Exception as e:
        print(f"❌ Dataset loading test failed: {e}")
        return False
    
    # Test inference
    try:
        results_df = analyzer.run_inference(test_loader)
        print(f"✅ Inference test passed! Analyzed {len(results_df)} samples.")
        print(f"   Columns: {list(results_df.columns)}")
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False
    
    # Test visualization functions (without showing plots)
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Test confidence vs steps plot
        bin_stats = analyzer.plot_confidence_vs_steps()
        print("✅ Confidence vs steps plot test passed!")
        
        # Test steps per class plot
        class_steps, pred_class_steps = analyzer.plot_steps_per_class()
        print("✅ Steps per class plot test passed!")
        
        # Test summary report
        report = analyzer.generate_summary_report()
        print("✅ Summary report test passed!")
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False
    
    # Cleanup
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Cleaned up dummy checkpoint: {checkpoint_path}")
    
    print("🎉 All analyzer tests passed!")
    return True

if __name__ == "__main__":
    test_analyzer()