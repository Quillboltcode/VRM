import argparse
import os
import gc
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from dataset import ImageFolderDataset, get_default_transform, create_imagefolder_loaders, create_test_loader, extract_subject_id
from loss import PonderLoss
from model import RecursiveFER
from train import train_single_fold, final_test_evaluation


def run_final_test_evaluation(args, device, cv_loaders=None):
    """
    Run final test evaluation after training for ImageFolder datasets.
    """
    try:
        print(f"\nCreating test loader for final evaluation...")
        
        # Create test loader - ImageFolder always
        test_loader, test_dataset = create_test_loader(
            root_dir=args.root_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
            use_imagefolder=True,
            label_file=None
        )
        
        print(f"Test dataset size: {len(test_dataset)} samples")
        
        # Debug: Print test dataset class info
        if hasattr(test_dataset, 'full_dataset'):
            num_classes = len(test_dataset.full_dataset.classes)
            class_names = test_dataset.full_dataset.classes
            print(f"DEBUG: Test dataset has {num_classes} classes: {class_names}")
        else:
            print(f"DEBUG: Could not determine test dataset classes")
        
        # Find all saved models
        model_paths = []
        
        # Find models in checkpoints directory
        import glob
        model_paths = glob.glob("./checkpoints/best_model_fold_*.pth")
        
        if not model_paths:
            print("Warning: No saved models found for final evaluation.")
            return
        
        print(f"Found {len(model_paths)} saved models for evaluation")
        
        # Run final evaluation
        final_test_evaluation(model_paths, test_loader, device, test_dataset)
        
    except Exception as e:
        print(f"Error during final test evaluation: {e}")
        import traceback
        traceback.print_exc()


# Kaggle specific imports. These will fail if not in a Kaggle environment.
try:
    from kaggle_secrets import UserSecretsClient
except ImportError:
    UserSecretsClient = None


def train_imagefolder_cv(args):
    """
    Train RecursiveFER model with ImageFolder dataset using GroupKFold cross-validation.
    """
    # Wandb setup
    wandb_enabled = True
    if UserSecretsClient is not None:
        try:
            user_secrets = UserSecretsClient()
            wandb_api_key = user_secrets.get_secret("wandb_api_secret")
            wandb.login(key=wandb_api_key)
            print("Wandb login successful")
        except Exception as e:
            print(f"Wandb login failed: {e}")
            wandb_enabled = False
    else:
        print("Wandb API key not found. Disabling wandb logging.")
        wandb_enabled = False

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data paths - ImageFolder specific
    root_dir = args.root_dir
    
    # Check if the path contains a 'train' subdirectory (common in Kaggle datasets)
    # If so, point to that instead of the parent to ensure ImageFolder finds the correct classes
    if os.path.isdir(os.path.join(root_dir, "train")):
        print(f"Detected 'train' subdirectory. Updating root_dir from '{root_dir}' to '{os.path.join(root_dir, 'train')}'")
        root_dir = os.path.join(root_dir, "train")
        
    print(f"Using ImageFolder dataset from: {root_dir}")

    # Create balanced loaders with cross-validation
    print("Creating balanced data loaders with cross-validation...")
    try:
        try:
            cv_loaders = create_imagefolder_loaders(
                root_dir=root_dir,
                batch_size=args.batch_size,
                image_size=args.image_size,
                num_workers=args.num_workers,
                use_weighted_sampler=args.use_weighted_sampler,
                n_splits=args.n_folds,
                random_state=args.random_state,
                train_split=args.train_split
            )
            
            # Debug: Print number of classes detected
            if cv_loaders:
                first_loader = cv_loaders[0]['train_loader']
                if hasattr(first_loader.dataset, 'full_dataset'):
                    num_classes = len(first_loader.dataset.full_dataset.classes)
                    class_names = first_loader.dataset.full_dataset.classes
                    print(f"DEBUG: Detected {num_classes} classes: {class_names}")
                else:
                    print(f"DEBUG: Could not determine number of classes")
        except ValueError as e:
            if "Cannot have number of splits" in str(e):
                print(f"Warning: {e}")
                print(f"Falling back to {min(args.n_folds, len(set([extract_subject_id(f) for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])))} splits based on available groups...")
                # Fall back to fewer splits
                available_groups = len(set([extract_subject_id(f) for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]))
                actual_splits = min(args.n_folds, max(1, available_groups))
                cv_loaders = create_imagefolder_loaders(
                    root_dir=root_dir,
                    batch_size=args.batch_size,
                    image_size=args.image_size,
                    num_workers=args.num_workers,
                    use_weighted_sampler=args.use_weighted_sampler,
                    n_splits=actual_splits,
                    random_state=args.random_state,
                    train_split=args.train_split
                )
            else:
                raise e
    
        if args.single_fold is not None:
            # Train on single specific fold
            if args.single_fold >= len(cv_loaders):
                print(f"Error: Fold {args.single_fold} not found. Available folds: 0-{len(cv_loaders)-1}")
                print(f"Total folds created: {len(cv_loaders)}")
                for i, fold_data in enumerate(cv_loaders):
                    train_size = len(fold_data['train_loader'].dataset)
                    val_size = len(fold_data['val_loader'].dataset)
                    print(f"  Fold {i}: {train_size} train samples, {val_size} val samples")
                return
            print(f"Training on single fold {args.single_fold}...")
            print(f"Fold {args.single_fold} details:")
            train_size = len(cv_loaders[args.single_fold]['train_loader'].dataset)
            val_size = len(cv_loaders[args.single_fold]['val_loader'].dataset)
            print(f"  Train samples: {train_size}")
            print(f"  Val samples: {val_size}")
            
            fold_data = cv_loaders[args.single_fold]
            best_acc = train_single_fold(fold_data, args, device, wandb_enabled)
            print(f"\nFinal validation accuracy for fold {args.single_fold}: {best_acc:.4f}")
            
            # Final test evaluation for single fold
            if args.run_final_test:
                run_final_test_evaluation(args, device, [fold_data])
            
        elif args.use_cross_validation:
            print(f"Training with {args.n_folds}-fold cross-validation...")
            fold_accuracies = []
            
            for fold_data in cv_loaders:
                best_acc = train_single_fold(fold_data, args, device, wandb_enabled)
                fold_accuracies.append(best_acc)
            
            # Print cross-validation results
            print(f"\n{'='*50}")
            print("CROSS-VALIDATION RESULTS")
            print(f"{'='*50}")
            for i, acc in enumerate(fold_accuracies):
                print(f"Fold {i+1}: {acc:.4f}")
            print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
            print(f"{'='*50}")
            
            # Final test evaluation after cross-validation
            if args.run_final_test:
                run_final_test_evaluation(args, device, cv_loaders)
            
        else:
            # Train on single split (first fold only)
            print("Training on single split...")
            fold_data = cv_loaders[0]
            best_acc = train_single_fold(fold_data, args, device, wandb_enabled)
            print(f"\nFinal validation accuracy: {best_acc:.4f}")
            
            # Final test evaluation for single fold
            if args.run_final_test:
                run_final_test_evaluation(args, device, [fold_data])
                
    except Exception as e:
        print(f"Error creating balanced loaders: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup memory to avoid OOM in fallback
        if 'cv_loaders' in locals():
            del cv_loaders
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("Falling back to simple training setup...")
        
        # Fallback to original simple training
        train_transform = get_default_transform(image_size=args.image_size, is_train=True)
        test_transform = get_default_transform(image_size=args.image_size, is_train=False)

        train_dataset = ImageFolderDataset(root_dir=root_dir, transform=train_transform, is_train=True, train_split=args.train_split)
        test_dataset = ImageFolderDataset(root_dir=root_dir, transform=test_transform, is_train=False, train_split=args.train_split)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        # Model, Loss, Optimizer - detect number of classes from dataset
        if hasattr(train_dataset, 'class_to_idx'):
            num_classes = len(train_dataset.class_to_idx)
        else:
            num_classes = 7
        
        model = RecursiveFER(in_channels=3, num_classes=num_classes, hidden_dim=args.hidden_dim, max_steps=args.max_steps).to(device)
        classification_loss = torch.nn.CrossEntropyLoss()
        criterion = PonderLoss(classification_loss, lambda_ponder=args.lambda_ponder)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

        # Wandb setup for fallback
        if wandb_enabled:
            try:
                wandb.init(
                    project="trm-fer-act", 
                    config=args,
                    name="fallback_simple_train",
                    reinit=True
                )
                wandb.watch(model)
            except Exception as e:
                print(f"Warning: Wandb initialization failed: {e}")
                wandb_enabled = False

        # Simple training loop
        for epoch in range(args.epochs):
            model.train()
            total_loss, total_class_loss, total_ponder_cost, total_steps = 0, 0, 0, 0
            
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                logits, num_steps = model(images)
                loss, class_loss, ponder_cost = criterion(logits, labels, num_steps)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_class_loss += class_loss.item()
                total_ponder_cost += ponder_cost.item()
                total_steps += num_steps.mean().item()

            avg_loss = total_loss / len(train_loader)
            avg_steps = total_steps / len(train_loader)

            # Validation
            from train import evaluate
            val_acc, val_avg_steps = evaluate(model, test_loader, device)
            
            if wandb_enabled:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "train_avg_steps": avg_steps,
                    "val_accuracy": val_acc,
                    "val_avg_steps": val_avg_steps,
                })
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}, Avg Steps: {val_avg_steps:.2f}")

            # Use specified checkpoint directory or default to ./checkpoints
            if args.checkpoint_dir:
                checkpoint_dir = args.checkpoint_dir
                os.makedirs(checkpoint_dir, exist_ok=True)
                output_dir = checkpoint_dir
            else:
                os.makedirs("./checkpoints", exist_ok=True)
                output_dir = "./checkpoints"
            torch.save(model.state_dict(), f"./checkpoints/model_epoch_{epoch+1}.pth")
            
        if wandb_enabled:
            try:
                wandb.finish()
            except:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RecursiveFER model with ImageFolder dataset and GroupKFold cross-validation.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension of model")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of recursive steps")
    parser.add_argument("--lambda_ponder", type=float, default=0.01, help="Lambda for ponder cost")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--use_cross_validation", action="store_true", help="Use cross-validation instead of single split")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--single_fold", type=int, default=None, help="Train on a single specific fold (0-based)")
    parser.add_argument("--use_weighted_sampler", action="store_true", help="Use weighted sampler for class balancing")
    parser.add_argument("--train_split", type=float, default=0.8, help="Train/validation split ratio for simple training")
        parser.add_argument("--root_dir", type=str, default="/kaggle/input/rafdb", help="Root directory of the dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="/kaggle/working/checkpoint", help="Directory to save model checkpoints")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--run_final_test", action="store_true", help="Run final evaluation on test set after training")
    
    args = parser.parse_args()
    
    train_imagefolder_cv(args)