#!/usr/bin/env python3
"""
Test script to verify dataset functionality and fixes.
Creates dummy ImageFolder and RAF-DB datasets to test both implementations.
"""

import os
import tempfile
import shutil
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

# Import our dataset classes
from dataset import (
    ImageFolderDataset, 
    RafDBDataset,
    create_imagefolder_loaders,
    create_balanced_loaders,
    create_group_kfold_splits,
    extract_subject_id,
    get_default_transform
)


def create_dummy_imagefolder_dataset(temp_dir, num_classes=3, samples_per_class=10):
    """Create a dummy ImageFolder dataset for testing."""
    classes = [f'class_{i}' for i in range(num_classes)]
    
    for class_name in classes:
        class_dir = os.path.join(temp_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for i in range(samples_per_class):
            # Create images with different patterns
            if i % 3 == 0:
                filename = f"subject{class_name}_{i:03d}.jpg"
            elif i % 3 == 1:
                filename = f"{class_name}_{i:03d}_aligned.jpg"
            else:
                filename = f"train_{i:03d}.jpg"
            
            # Create different colored images
            color = (i * 25 % 256, (i * 50) % 256, (i * 75) % 256)
            img = Image.new('RGB', (64, 64), color)
            img.save(os.path.join(class_dir, filename))
    
    return temp_dir


def create_dummy_rafdb_dataset(temp_dir, num_samples=30):
    """Create a dummy RAF-DB dataset for testing."""
    # Create label file
    label_file = os.path.join(temp_dir, "list_patition_label.txt")
    
    with open(label_file, "w") as f:
        for i in range(num_samples):
            # Split between train and test
            if i % 2 == 0:
                prefix = "train"
            else:
                prefix = "test"
            
            # Subject-based naming
            subject_id = f"{(i // 3) + 1:04d}"
            filename = f"{prefix}_{subject_id}_aligned.jpg"
            label = (i % 7) + 1  # RAF-DB has 7 emotion classes
            
            f.write(f"{filename} {label}\n")
    
    # Create image directory
    img_dir = os.path.join(temp_dir, "Image", "aligned")
    os.makedirs(img_dir, exist_ok=True)
    
    # Create dummy images
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            filename = parts[0]
            
            color = (hash(filename) % 256, (hash(filename) * 2) % 256, (hash(filename) * 3) % 256)
            img = Image.new('RGB', (64, 64), color)
            img.save(os.path.join(img_dir, filename))
    
    return temp_dir, label_file


def test_extract_subject_id():
    """Test the flexible subject ID extraction."""
    print("Testing subject ID extraction...")
    
    test_cases = [
        ("train_0001_aligned.jpg", "0001"),
        ("test_0002_aligned.jpg", "0002"),
        ("subject001_img01.jpg", "001"),
        ("subject_002_01.jpg", "002"),
        ("001_image_01.jpg", "001"),
        ("random_name.jpg", "random"),
    ]
    
    for filename, expected in test_cases:
        result = extract_subject_id(filename)
        print(f"  {filename} -> {result} (expected: {expected})")
        assert result == expected, f"Mismatch for {filename}: got {result}, expected {expected}"
    
    print("✓ Subject ID extraction tests passed!\n")


def test_imagefolder_dataset():
    """Test ImageFolderDataset functionality."""
    print("Testing ImageFolderDataset...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy dataset
        dataset_dir = create_dummy_imagefolder_dataset(temp_dir)
        
        # Test with split
        transform = get_default_transform(image_size=64, is_train=True)
        train_dataset = ImageFolderDataset(dataset_dir, transform=transform, is_train=True, random_state=42)
        test_dataset = ImageFolderDataset(dataset_dir, transform=transform, is_train=False, random_state=42)
        
        print(f"  Train dataset size: {len(train_dataset)}")
        print(f"  Test dataset size: {len(test_dataset)}")
        
        # Test data loading
        image, label = train_dataset[0]
        assert isinstance(image, torch.Tensor), f"Expected tensor, got {type(image)}"
        assert isinstance(label, int), f"Expected int, got {type(label)}"
        
        # Test with full dataset
        full_dataset = ImageFolderDataset(dataset_dir, use_full_dataset=True)
        print(f"  Full dataset size: {len(full_dataset)}")
        
        print("✓ ImageFolderDataset tests passed!\n")


def test_rafdb_dataset():
    """Test RafDBDataset functionality."""
    print("Testing RafDBDataset...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy dataset
        dataset_dir, label_file = create_dummy_rafdb_dataset(temp_dir)
        
        # Test datasets
        transform = get_default_transform(image_size=64, is_train=True)
        train_dataset = RafDBDataset(dataset_dir, label_file, transform=transform, is_train=True)
        test_dataset = RafDBDataset(dataset_dir, label_file, transform=transform, is_train=False)
        
        print(f"  Train dataset size: {len(train_dataset)}")
        print(f"  Test dataset size: {len(test_dataset)}")
        
        # Test data loading
        image, label = train_dataset[0]
        assert isinstance(image, torch.Tensor), f"Expected tensor, got {type(image)}"
        assert isinstance(label, int), f"Expected int, got {type(label)}"
        
        print("✓ RafDBDataset tests passed!\n")


def test_cross_validation():
    """Test cross-validation functionality."""
    print("Testing cross-validation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test ImageFolder cross-validation
        print("  Testing ImageFolder cross-validation...")
        dataset_dir = create_dummy_imagefolder_dataset(temp_dir, num_classes=3, samples_per_class=20)
        
        try:
            loaders = create_imagefolder_loaders(
                dataset_dir, 
                batch_size=4, 
                image_size=64, 
                n_splits=3,
                random_state=42
            )
            
            print(f"    Created {len(loaders)} folds")
            
            for i, fold_data in enumerate(loaders):
                train_loader = fold_data['train_loader']
                val_loader = fold_data['val_loader']
                
                print(f"    Fold {i}: {len(train_loader)} train batches, {len(val_loader)} val batches")
                
                # Test data loading from first batch
                train_images, train_labels = next(iter(train_loader))
                assert train_images.shape[0] <= 4, f"Batch size mismatch: {train_images.shape}"
                assert train_labels.shape[0] <= 4, f"Label batch size mismatch: {train_labels.shape}"
                
        except Exception as e:
            print(f"    ✗ ImageFolder cross-validation failed: {e}")
            return False
        
        # Test RAF-DB cross-validation
        print("  Testing RAF-DB cross-validation...")
        dataset_dir, label_file = create_dummy_rafdb_dataset(temp_dir)
        
        try:
            loaders = create_balanced_loaders(
                dataset_dir, 
                label_file,
                batch_size=4, 
                image_size=64, 
                n_splits=3,
                random_state=42
            )
            
            print(f"    Created {len(loaders)} folds")
            
            for i, fold_data in enumerate(loaders):
                train_loader = fold_data['train_loader']
                val_loader = fold_data['val_loader']
                
                print(f"    Fold {i}: {len(train_loader)} train batches, {len(val_loader)} val batches")
                
                # Test data loading from first batch
                train_images, train_labels = next(iter(train_loader))
                assert train_images.shape[0] <= 4, f"Batch size mismatch: {train_images.shape}"
                assert train_labels.shape[0] <= 4, f"Label batch size mismatch: {train_labels.shape}"
                
        except Exception as e:
            print(f"    ✗ RAF-DB cross-validation failed: {e}")
            return False
    
    print("✓ Cross-validation tests passed!\n")
    return True


def test_reproducibility():
    """Test that results are reproducible with same random seed."""
    print("Testing reproducibility...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_dir = create_dummy_imagefolder_dataset(temp_dir, num_classes=3, samples_per_class=15)
        
        # Create loaders twice with same seed
        loaders1 = create_imagefolder_loaders(
            dataset_dir, batch_size=4, image_size=64, n_splits=2, random_state=42
        )
        loaders2 = create_imagefolder_loaders(
            dataset_dir, batch_size=4, image_size=64, n_splits=2, random_state=42
        )
        
        # Check if splits are the same
        for i in range(len(loaders1)):
            subjects1 = loaders1[i]['train_subjects']
            subjects2 = loaders2[i]['train_subjects']
            
            assert np.array_equal(subjects1, subjects2), f"Fold {i} subjects don't match"
        
        print("✓ Reproducibility tests passed!\n")
        return True


def main():
    """Run all tests."""
    print("🧪 Running Dataset Tests\n")
    
    tests = [
        test_extract_subject_id,
        test_imagefolder_dataset,
        test_rafdb_dataset,
        test_cross_validation,
        test_reproducibility,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}\n")
            failed += 1
    
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Dataset functionality is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)