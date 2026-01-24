#!/usr/bin/env python3
"""
Test script to verify class balancing and cross-validation functionality.
"""

import os
import tempfile
import shutil
import numpy as np
from torch.utils.data import DataLoader

# Test the new functions
from dataset import (
    RafDBDataset, 
    get_default_transform, 
    get_class_weights,
    create_weighted_sampler,
    create_group_kfold_splits,
    create_balanced_loaders
)

def create_test_dataset():
    """Create a test dataset for evaluation."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create label file with imbalanced classes
    label_file = os.path.join(temp_dir, "labels.txt")
    with open(label_file, "w") as f:
        # Create imbalanced dataset: class 0: 20 samples, class 1: 5 samples, class 2: 3 samples
        for i in range(20):
            f.write(f"train_{i+1:03d}_aligned.jpg 1\n")  # Class 0 (Surprise)
        for i in range(5):
            f.write(f"train_{i+21:03d}_aligned.jpg 2\n")  # Class 1 (Fear) 
        for i in range(3):
            f.write(f"train_{i+26:03d}_aligned.jpg 3\n")  # Class 2 (Disgust)
    
    # Create image directory
    img_dir = os.path.join(temp_dir, "Image", "aligned")
    os.makedirs(img_dir, exist_ok=True)
    
    # Create dummy images
    from PIL import Image
    for i in range(28):
        Image.new('RGB', (100, 100)).save(os.path.join(img_dir, f"train_{i+1:03d}_aligned.jpg"))
    
    return temp_dir, label_file

def test_class_weights():
    """Test class weight calculation."""
    print("Testing class weight calculation...")
    
    # Create imbalanced labels
    labels = np.array([0]*20 + [1]*5 + [2]*3)  # Imbalanced: 20, 5, 3
    
    weights = get_class_weights(labels)
    
    print(f"Label counts: {np.bincount(labels)}")
    print(f"Class weights: {weights}")
    
    # Verify weights are inversely proportional to counts
    assert len(weights) == 3, f"Expected 3 weights, got {len(weights)}"
    assert weights[0] < weights[1] < weights[2], "Weights should increase as class frequency decreases"
    
    print("✅ Class weight calculation test passed!")
    return True

def test_weighted_sampler():
    """Test weighted sampler creation."""
    print("\nTesting weighted sampler...")
    
    # Create imbalanced labels
    labels = np.array([0]*20 + [1]*5 + [2]*3)
    
    sampler = create_weighted_sampler(labels)
    
    print(f"Sampler created successfully")
    print(f"Sampler type: {type(sampler)}")
    
    # Test sampling distribution
    sample_counts = {0: 0, 1: 0, 2: 0}
    for i in range(1000):
        idx = next(iter(sampler))
        sample_counts[labels[idx]] += 1
    
    print(f"Sample distribution over 1000 draws: {sample_counts}")
    
    # Verify more balanced distribution
    ratio = max(sample_counts.values()) / min(sample_counts.values())
    assert ratio < 3.0, f"Sampling should be more balanced, ratio: {ratio:.2f}"
    
    print("✅ Weighted sampler test passed!")
    return True

def test_group_kfold():
    """Test GroupKFold split creation."""
    print("\nTesting GroupKFold splits...")
    
    temp_dir, label_file = create_test_dataset()
    
    try:
        dataset = RafDBDataset(temp_dir, label_file, transform=None, is_train=True)
        splits = create_group_kfold_splits(dataset, n_splits=3, random_state=42)
        
        print(f"Created {len(splits)} splits")
        
        for split in splits:
            print(f"Fold {split['fold']}:")
            print(f"  Train samples: {len(split['train_indices'])}")
            print(f"  Val samples: {len(split['val_indices'])}")
            print(f"  Train subjects: {len(set(split['train_subjects']))}")
            print(f"  Val subjects: {len(set(split['val_subjects']))}")
            
            # Verify no subject overlap
            train_subjects = set(split['train_subjects'])
            val_subjects = set(split['val_subjects'])
            overlap = train_subjects.intersection(val_subjects)
            assert len(overlap) == 0, f"Subject overlap detected: {overlap}"
        
        print("✅ GroupKFold split test passed!")
        return True
        
    finally:
        shutil.rmtree(temp_dir)

def test_balanced_loaders():
    """Test balanced loader creation."""
    print("\nTesting balanced loaders...")
    
    temp_dir, label_file = create_test_dataset()
    
    try:
        loaders = create_balanced_loaders(
            root_dir=temp_dir,
            label_file=label_file,
            batch_size=8,
            image_size=224,
            num_workers=0,  # Disable multiprocessing for testing
            use_weighted_sampler=True,
            n_splits=2,
            random_state=42
        )
        
        print(f"Created {len(loaders)} loaders")
        
        for loader_data in loaders:
            fold = loader_data['fold']
            train_loader = loader_data['train_loader']
            val_loader = loader_data['val_loader']
            class_weights = loader_data['class_weights']
            
            print(f"Fold {fold}:")
            print(f"  Train batches: {len(train_loader)}")
            print(f"  Val batches: {len(val_loader)}")
            print(f"  Class weights: {class_weights}")
            
            # Test data loading
            for images, labels in train_loader:
                print(f"    Train batch: images {images.shape}, labels {labels.shape}")
                break
                
            for images, labels in val_loader:
                print(f"    Val batch: images {images.shape}, labels {labels.shape}")
                break
        
        print("✅ Balanced loaders test passed!")
        return True
        
    finally:
        shutil.rmtree(temp_dir)

def main():
    """Run all tests."""
    print("Testing class balancing and cross-validation functionality...\n")
    
    tests = [
        test_class_weights,
        test_weighted_sampler,
        test_group_kfold,
        test_balanced_loaders
    ]
    
    passed = 0
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} tests passed")
    print(f"{'='*50}")
    
    if passed == len(tests):
        print("🎉 All tests passed!")
    else:
        print("💥 Some tests failed!")

if __name__ == "__main__":
    main()