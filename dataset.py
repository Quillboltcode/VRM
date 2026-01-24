
import os
from typing import Callable, Tuple, Optional

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import GroupKFold
from torchvision.datasets import ImageFolder


# Emotion mapping for RAF-DB (1-indexed in file)
# 1: Surprise, 2: Fear, 3: Disgust, 4: Happiness, 5: Sadness, 6: Anger, 7: Neutral
EMOTION_MAPPING = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happiness",
    5: "Sadness",
    6: "Anger",
    7: "Neutral",
}


class ImageFolderDataset(Dataset):
    """ImageFolder Compatible Dataset for GKFold and balanced sampling"""

    def __init__(
        self,
        root_dir: str,
        transform: Callable = None,
        is_train: bool = True,
        train_split: float = 0.8,
    ):
        """
        Args:
            root_dir (string): Directory with ImageFolder structure (class_name subdirs).
            transform (callable, optional): Optional transform to be applied on a sample.
            is_train (bool): If true, loads training data, otherwise test data.
            train_split (float): Fraction of data to use for training (0.0-1.0).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.train_split = train_split
        
        # Load ImageFolder dataset
        self.full_dataset = ImageFolder(root_dir, transform=None)
        
        # Create train/val split
        total_size = len(self.full_dataset)
        train_size = int(total_size * train_split)
        val_size = total_size - train_size
        
        # Generate indices
        indices = list(range(total_size))
        np.random.shuffle(indices)
        
        if is_train:
            self.indices = indices[:train_size]
        else:
            self.indices = indices[train_size:]
        
        # Extract samples and labels for this split
        self.samples = [self.full_dataset.samples[i] for i in self.indices]
        self.targets = [self.full_dataset.targets[i] for i in self.indices]
        self.class_to_idx = self.full_dataset.class_to_idx
        
        # Create DataFrame for compatibility with existing functions
        self.image_files = pd.DataFrame({
            'name': [os.path.basename(sample[0]) for sample in self.samples],
            'label': self.targets
        })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


class RafDBDataset(Dataset):
    """RAF-DB Custom Dataset (kept for backward compatibility)"""

    def __init__(
        self,
        root_dir: str,
        label_file: str,
        transform: Callable = None,
        is_train: bool = True,
    ):
        """
        Args:
            root_dir (string): Directory with all the images.
            label_file (string): Path to the file with labels.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_train (bool): If true, loads training data, otherwise test data.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train

        label_data = pd.read_csv(label_file, sep=" ", header=None)
        label_data.columns = ["name", "label"]

        if is_train:
            self.image_files = label_data[label_data["name"].str.startswith("train_")]
        else:
            self.image_files = label_data[label_data["name"].str.startswith("test_")]

        # Adjust labels to be 0-indexed
        self.image_files["label"] = self.image_files["label"] - 1

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if isinstance(idx, list):
            idx = idx[0] if idx else 0

        img_name = os.path.join(
            self.root_dir, "Image/aligned", self.image_files.iloc[idx, 0]
        )
        image = np.array(Image.open(img_name).convert("RGB"))
        label = int(self.image_files.iloc[idx, 1])

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

def get_default_transform(image_size: int = 224, is_train: bool = True):
    """
    Returns a default set of transforms for the dataset using albumentations.
    """
    # Fix normalization parameters to be tuples
    normalize_mean = (0.485, 0.456, 0.406)
    normalize_std = (0.229, 0.224, 0.225)
    
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.5),
            A.Normalize(mean=normalize_mean, std=normalize_std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=normalize_mean, std=normalize_std),
            ToTensorV2(),
        ])


def get_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Calculate class weights for imbalanced dataset.
    Returns weights inversely proportional to class frequencies.
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    
    # Create weight tensor mapping
    weight_tensor = torch.zeros(len(classes))
    for i, cls in enumerate(classes):
        weight_tensor[i] = weights[i]
    
    return weight_tensor


def create_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler for balanced sampling.
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=classes, y=labels)
    
    # Map each sample to its class weight
    sample_weights = np.array([class_weights[label] for label in labels])
    
    # Create sampler - convert to list for compatibility
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


def create_group_kfold_splits(dataset, n_splits: int = 5, random_state: int = 42):
    """
    Create GroupKFold splits for cross-validation.
    Groups are created based on subject IDs extracted from filenames.
    """
    # Extract subject IDs from filenames (assuming format: train_testsubject_XXX.jpg)
    subject_ids = []
    for idx in range(len(dataset)):
        filename = dataset.image_files.iloc[idx, 0]
        # Extract subject ID - look for patterns like "train_0001" -> subject "0001"
        parts = filename.split('_')
        if len(parts) >= 2:
            subject_id = parts[1]  # Second part should be subject ID
        else:
            subject_id = filename  # Fallback to filename
        
        subject_ids.append(subject_id)
    
        subject_ids = np.array(subject_ids)
        labels = np.array(dataset.image_files['label'])
    
    # Create GroupKFold splitter
    gkf = GroupKFold(n_splits=n_splits)
    
    splits = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(range(len(dataset)), labels, groups=subject_ids)):
        splits.append({
            'fold': fold,
            'train_indices': train_idx,
            'val_indices': val_idx,
            'train_subjects': subject_ids[train_idx],
            'val_subjects': subject_ids[val_idx]
        })
    
    return splits


def create_imagefolder_loaders(
    root_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
    n_splits: int = 5,
    random_state: int = 42,
    train_split: float = 0.8
):
    """
    Create balanced data loaders for ImageFolder datasets with cross-validation support.
    """
    # Create full dataset
    train_transform = get_default_transform(image_size=image_size, is_train=True)
    test_transform = get_default_transform(image_size=image_size, is_train=False)
    
    full_train_dataset = ImageFolderDataset(root_dir, transform=train_transform, is_train=True, train_split=train_split)
    full_test_dataset = ImageFolderDataset(root_dir, transform=test_transform, is_train=False, train_split=train_split)
    
    # Get class weights for loss function
    train_labels = np.array(full_train_dataset.image_files['label'])
    class_weights = get_class_weights(train_labels)
    
    # Create cross-validation splits
    cv_splits = create_group_kfold_splits(full_train_dataset, n_splits=n_splits, random_state=random_state)
    
    loaders = []
    for split in cv_splits:
        train_indices = split['train_indices']
        val_indices = split['val_indices']
        
        train_labels_subset = train_labels[train_indices]
        
        if use_weighted_sampler:
            train_sampler = create_weighted_sampler(train_labels_subset)
            train_shuffle = False
        else:
            train_sampler = None
            train_shuffle = True
        
        train_loader = DataLoader(
            full_train_dataset,
            batch_size=batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            drop_last=True
        )
        
        val_loader = DataLoader(
            full_test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False
        )
        
        loaders.append({
            'fold': split['fold'],
            'train_loader': train_loader,
            'val_loader': val_loader,
            'train_subjects': split['train_subjects'],
            'val_subjects': split['val_subjects'],
            'class_weights': class_weights
        })
    
    return loaders


def create_balanced_loaders(
    root_dir: str,
    label_file: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
    n_splits: int = 5,
    random_state: int = 42
):
    """
    Create balanced data loaders with cross-validation support.
    """
    # Create full dataset
    train_transform = get_default_transform(image_size=image_size, is_train=True)
    test_transform = get_default_transform(image_size=image_size, is_train=False)
    
    full_train_dataset = RafDBDataset(root_dir, label_file, transform=train_transform, is_train=True)
    full_test_dataset = RafDBDataset(root_dir, label_file, transform=test_transform, is_train=False)
    
    # Get class weights for loss function
    train_labels = np.array(full_train_dataset.image_files['label'])
    class_weights = get_class_weights(train_labels)
    
    # Create cross-validation splits
    cv_splits = create_group_kfold_splits(full_train_dataset, n_splits=n_splits, random_state=random_state)
    
    loaders = []
    for split in cv_splits:
        # Create subset datasets for this fold
        train_indices = split['train_indices']
        val_indices = split['val_indices']
        
        # Note: In practice, you'd create Subset datasets here
        # For now, we'll create samplers for the full datasets
        
        # Create samplers
        train_labels_subset = train_labels[train_indices]
        
        if use_weighted_sampler:
            train_sampler = create_weighted_sampler(train_labels_subset)
            train_shuffle = False  # Sampler handles ordering
        else:
            train_sampler = None
            train_shuffle = True
        
        # Create data loaders
        train_loader = DataLoader(
            full_train_dataset,
            batch_size=batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            drop_last=True
        )
        
        val_loader = DataLoader(
            full_test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False
        )
        
        loaders.append({
            'fold': split['fold'],
            'train_loader': train_loader,
            'val_loader': val_loader,
            'train_subjects': split['train_subjects'],
            'val_subjects': split['val_subjects'],
            'class_weights': class_weights
        })
    
    return loaders
