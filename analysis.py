
#!/usr/bin/env python3
"""
Analysis script for RecursiveFERModel model with Adaptive Computation Time.
Implements visualizations and analysis as specified in gemini.md.
"""

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RafDBDataset, ImageFolderDataset, get_default_transform, EMOTION_MAPPING
from model import RecursiveFER

# Kaggle specific imports
try:
    from kaggle_secrets import UserSecretsClient
except ImportError:
    UserSecretsClient = None


class ModelAnalyzer:
    """Class for analyzing trained RecursiveFER models."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.results_df = None
        
    def load_model(self, checkpoint_path: str = None) -> RecursiveFER:
        """Load a trained model checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path
            
        print(f"Loading model from {checkpoint_path}")
        
        # Initialize model with default parameters (adjust as needed)
        self.model = RecursiveFER(
            in_channels=3,
            num_classes=7,  # Will be updated based on checkpoint
            hidden_dim=128,
            max_steps=10
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Determine number of classes from checkpoint (for compatibility with different datasets)
        if 'num_classes' in checkpoint:
            num_classes = checkpoint['num_classes']
            print(f"DEBUG: Checkpoint has {num_classes} classes saved")
        else:
            # Try to determine from test dataset first, then fallback to 7
            if hasattr(self, 'results_df') and self.results_df is not None and len(self.results_df) > 0:
                # Check if we have test data with class information
                if 'predicted_class' in self.results_df.columns:
                    num_classes = self.results_df['predicted_class'].max() + 1
                    print(f"DEBUG: Determined {num_classes} classes from test predictions")
                elif hasattr(self.results_df, 'ground_truth_class'):
                    num_classes = self.results_df['ground_truth_class'].max() + 1
                    print(f"DEBUG: Determined {num_classes} classes from test ground truth")
                else:
                    num_classes = 7
                    print("DEBUG: Could not determine classes from test data, using default 7")
            else:
                num_classes = 7  # Default fallback
                print("DEBUG: No test data available, using default 7")
        
        self.model = RecursiveFER(
            in_channels=3,
            num_classes=num_classes,
            hidden_dim=128,
            max_steps=10
        )
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"DEBUG: Model initialized with {num_classes} output classes")
        
        print("Model loaded successfully!")
        return self.model
    
    def run_inference(self, dataloader: DataLoader) -> pd.DataFrame:
        """Run inference on dataset and collect analysis data."""
        if self.model is None:
            self.load_model()
            
        results = []
        
        print("Running inference...")
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Get model predictions and step counts
                logits, steps_taken, _ = self.model(images)  # Returns (logits, steps_taken, halt_probs)
                
                # Calculate confidence (softmax probability of predicted class)
                probs = F.softmax(logits, dim=1)
                confidence, predicted = probs.max(dim=1)
                
                # Store results for each sample in batch
                for i in range(images.size(0)):
                    results.append({
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'predicted_class': predicted[i].item(),
                        'ground_truth_class': labels[i].item(),
                        'confidence': confidence[i].item(),
                        'num_steps': steps_taken[i].item(),
                        'predicted_label': EMOTION_MAPPING[predicted[i].item() + 1],
                        'ground_truth_label': EMOTION_MAPPING[labels[i].item() + 1],
                        'is_correct': predicted[i].item() == labels[i].item()
                    })
        
        self.results_df = pd.DataFrame(results)
        print(f"Inference completed. Analyzed {len(results)} samples.")
        return self.results_df
    
    def plot_confidence_vs_steps(self, save_path: str = None):
        """Create scatter plot of confidence vs number of steps."""
        if self.results_df is None:
            raise ValueError("Run inference first!")
            
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot
        plt.subplot(2, 2, 1)
        sns.scatterplot(
            data=self.results_df,
            x='confidence',
            y='num_steps',
            alpha=0.6,
            hue='is_correct',
            palette={True: 'green', False: 'red'}
        )
        plt.title('Confidence vs Number of Steps')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Number of Steps')
        plt.legend(['Incorrect', 'Correct'])
        
        # Create binned analysis
        plt.subplot(2, 2, 2)
        self.results_df['confidence_bin'] = pd.cut(
            self.results_df['confidence'], 
            bins=10, 
            labels=[f'{i/10:.1f}-{(i+1)/10:.1f}' for i in range(10)]
        )
        
        bin_stats = self.results_df.groupby('confidence_bin').agg({
            'num_steps': 'mean',
            'confidence': 'count'
        }).rename(columns={'confidence': 'count'})
        
        plt.bar(range(len(bin_stats)), bin_stats['num_steps'])
        plt.xlabel('Confidence Bins')
        plt.ylabel('Average Steps')
        plt.title('Average Steps per Confidence Bin')
        plt.xticks(range(len(bin_stats)), bin_stats.index, rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return bin_stats
    
    def plot_steps_per_class(self, save_path: str = None):
        """Create histogram of average steps per emotion class."""
        if self.results_df is None:
            raise ValueError("Run inference first!")
            
        plt.figure(figsize=(14, 6))
        
        # Average steps per ground truth class
        plt.subplot(1, 2, 1)
        class_steps = self.results_df.groupby('ground_truth_label')['num_steps'].agg(['mean', 'std'])
        
        bars = plt.bar(range(len(class_steps)), class_steps['mean'])
        plt.errorbar(
            range(len(class_steps)), 
            class_steps['mean'], 
            yerr=class_steps['std'], 
            fmt='none', 
            color='black', 
            capsize=5
        )
        plt.xlabel('Emotion Class')
        plt.ylabel('Average Steps')
        plt.title('Average Steps per Ground Truth Emotion')
        plt.xticks(range(len(class_steps)), class_steps.index, rotation=45)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height() + class_steps['std'].iloc[i],
                f'{class_steps["mean"].iloc[i]:.2f}',
                ha='center', 
                va='bottom'
            )
        
        # Steps per predicted class
        plt.subplot(1, 2, 2)
        pred_class_steps = self.results_df.groupby('predicted_label')['num_steps'].agg(['mean', 'std'])
        
        bars = plt.bar(range(len(pred_class_steps)), pred_class_steps['mean'])
        plt.errorbar(
            range(len(pred_class_steps)), 
            pred_class_steps['mean'], 
            yerr=pred_class_steps['std'], 
            fmt='none', 
            color='black', 
            capsize=5
        )
        plt.xlabel('Emotion Class')
        plt.ylabel('Average Steps')
        plt.title('Average Steps per Predicted Emotion')
        plt.xticks(range(len(pred_class_steps)), pred_class_steps.index, rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return class_steps, pred_class_steps
    
    def visualize_extreme_examples(self, dataloader: DataLoader, save_dir: str = None):
        """Visualize images that took few vs many steps."""
        if self.results_df is None:
            raise ValueError("Run inference first!")
        
        # Find extreme examples
        min_steps = self.results_df['num_steps'].min()
        max_steps = self.results_df['num_steps'].max()
        
        min_step_examples = self.results_df[self.results_df['num_steps'] == min_steps].head(9)
        max_step_examples = self.results_df[self.results_df['num_steps'] == max_steps].head(9)
        
        def plot_examples(examples_df, title):
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            fig.suptitle(title, fontsize=16)
            
            for idx, (_, row) in enumerate(examples_df.iterrows()):
                if idx >= 9:
                    break
                    
                # Get the actual image from dataloader
                batch_idx = row['batch_idx']
                sample_idx = row['sample_idx']
                
                # Get the batch (need to recreate this - simplified for demo)
                for batch_images, batch_labels in dataloader:
                    if batch_images.shape[0] > sample_idx:
                        image = batch_images[sample_idx].permute(1, 2, 0).numpy()
                        # Denormalize
                        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                        image = np.clip(image, 0, 1)
                        break
                
                ax = axes[idx // 3, idx % 3]
                ax.imshow(image)
                ax.set_title(
                    f"Steps: {row['num_steps']:.0f}\n"
                    f"GT: {row['ground_truth_label']}\n"
                    f"Pred: {row['predicted_label']}\n"
                    f"Conf: {row['confidence']:.3f}\n"
                    f"Correct: {row['is_correct']}"
                )
                ax.axis('off')
            
            plt.tight_layout()
            if save_dir:
                plt.savefig(f"{save_dir}/{title.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        plot_examples(min_step_examples, f"Images with Minimum Steps ({min_steps:.0f})")
        plot_examples(max_step_examples, f"Images with Maximum Steps ({max_steps:.0f})")
    
    def generate_summary_report(self, save_path: str = None):
        """Generate a summary report of the analysis."""
        if self.results_df is None:
            raise ValueError("Run inference first!")
        
        report = {
            'Total Samples': len(self.results_df),
            'Accuracy': (self.results_df['is_correct'].mean() * 100),
            'Average Steps': self.results_df['num_steps'].mean(),
            'Steps Std': self.results_df['num_steps'].std(),
            'Min Steps': self.results_df['num_steps'].min(),
            'Max Steps': self.results_df['num_steps'].max(),
            'Average Confidence': self.results_df['confidence'].mean(),
        }
        
        # Correlation between confidence and steps
        correlation = self.results_df['confidence'].corr(self.results_df['num_steps'])
        report['Confidence-Steps Correlation'] = correlation
        
        # Steps for correct vs incorrect predictions
        correct_steps = self.results_df[self.results_df['is_correct']]['num_steps'].mean()
        incorrect_steps = self.results_df[~self.results_df['is_correct']]['num_steps'].mean()
        report['Avg Steps (Correct)'] = correct_steps
        report['Avg Steps (Incorrect)'] = incorrect_steps
        
        # Print report
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY REPORT")
        print("="*50)
        for key, value in report.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        print("="*50)
        
        # Save to file if requested
        if save_path:
            with open(save_path, 'w') as f:
                for key, value in report.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Analyze RecursiveFER model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_root", type=str, default="/kaggle/input/rafdb", help="Root directory for dataset")
    parser.add_argument("--use_imagefolder", action="store_true", help="Use ImageFolder dataset instead of RAF-DB")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="./analysis_output", help="Output directory for plots")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test dataset - support both RAF-DB and ImageFolder
    test_transform = get_default_transform(image_size=224, is_train=False)
    
    if args.use_imagefolder:
        # ImageFolder dataset - need to point to test folder
        test_root_dir = args.data_root
        # Check if there's a test subdirectory
        if os.path.isdir(os.path.join(args.data_root, "test")):
            test_root_dir = os.path.join(args.data_root, "test")
        
        test_dataset = ImageFolderDataset(root_dir=test_root_dir, transform=test_transform, use_full_dataset=True)
        print(f"Using ImageFolder test dataset from: {test_root_dir}")
        # Debug: Print number of classes detected
        if hasattr(test_dataset, 'full_dataset'):
            num_classes = len(test_dataset.full_dataset.classes)
            class_names = test_dataset.full_dataset.classes
            print(f"DEBUG: ImageFolder test dataset has {num_classes} classes: {class_names}")
        else:
            print(f"DEBUG: Could not determine ImageFolder test dataset classes")
    else:
        # RAF-DB dataset
        label_file = os.path.join(args.data_root, "EmoLabel/list_patition_label.txt")
        test_dataset = RafDBDataset(args.data_root, label_file, transform=test_transform, is_train=False)
        print(f"Using RAF-DB test dataset from: {args.data_root}")
        # Debug: Print number of classes detected
        if hasattr(test_dataset, 'image_files'):
            unique_labels = test_dataset.image_files['label'].unique()
            num_classes = len(unique_labels)
            print(f"DEBUG: RAF-DB test dataset has {num_classes} classes: {sorted(unique_labels)}")
        else:
            print(f"DEBUG: Could not determine RAF-DB test dataset classes")
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize analyzer
    analyzer = ModelAnalyzer(args.checkpoint, args.device)
    
    # Run inference and collect data
    results_df = analyzer.run_inference(test_loader)
    
    # Generate visualizations
    print("Generating confidence vs steps plot...")
    analyzer.plot_confidence_vs_steps(os.path.join(args.output_dir, "confidence_vs_steps.png"))
    
    print("Generating steps per class plot...")
    analyzer.plot_steps_per_class(os.path.join(args.output_dir, "steps_per_class.png"))
    
    print("Generating extreme examples visualization...")
    analyzer.visualize_extreme_examples(test_loader, args.output_dir)
    
    # Generate summary report
    print("Generating summary report...")
    report = analyzer.generate_summary_report(os.path.join(args.output_dir, "summary_report.txt"))
    
    print(f"Analysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
