import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from dataset import ImageFolderDataset, get_default_transform, create_imagefolder_loaders, create_test_loader, extract_subject_id
from loss import PonderLoss
from model import RecursiveFERModel as RecursiveFER


def train_single_fold(fold_data, args, device, wandb_enabled=True):
    """Simplified train model on a single fold."""
    fold_num = fold_data['fold']
    train_loader = fold_data['train_loader']
    val_loader = fold_data['val_loader']
    class_weights = fold_data['class_weights']
    
    print(f"Training Fold {fold_num + 1}")
    
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    num_classes = len(class_weights)
    model = RecursiveFER(
        in_channels=3, 
        num_classes=num_classes, 
        hidden_dim=args.hidden_dim, 
        max_steps=args.max_steps
    ).to(device)
    
    classification_loss = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    criterion = PonderLoss(classification_loss, lambda_ponder=args.lambda_ponder)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    if wandb_enabled:
        try:
            wandb.init(
                project="trm-fer-act", 
                config=vars(args),
                name=f"run-fold_{fold_num}",
                dir="/kaggle/working/VRM/",
                reinit=True
            )
            wandb.watch(model)
        except:
            wandb_enabled = False

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss, total_class_loss, total_ponder_cost, total_steps = 0, 0, 0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Fold {fold_num+1} Epoch {epoch+1}/{args.epochs}"):
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
        avg_class_loss = total_class_loss / len(train_loader)
        avg_ponder_cost = total_ponder_cost / len(train_loader)
        avg_steps = total_steps / len(train_loader)

        val_acc, val_avg_steps = evaluate(model, val_loader, device)

        if wandb_enabled:
            wandb.log({
                "epoch": epoch,
                "fold": fold_num,
                "train_loss": avg_loss,
                "val_accuracy": val_acc,
                "val_avg_steps": val_avg_steps,
                "class_loss": avg_class_loss,
                "ponder_cost": avg_ponder_cost,
                "avg_steps": avg_steps
            })

        print(f"Fold {fold_num+1} Epoch {epoch+1}: Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if wandb_enabled and wandb.run:
                output_dir = wandb.run.dir
                wandb.save(os.path.join(output_dir, f"best_model_fold_{fold_num}.pth"))
            else:
                output_dir = "./checkpoints"
                os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, f"best_model_fold_{fold_num}.pth"))

    if wandb_enabled:
        try:
            wandb.finish()
        except:
            pass
    
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return best_val_acc


def evaluate(model, dataloader, device):
    """Simple evaluation function."""
    model.eval()
    correct = 0
    total = 0
    total_steps = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits, num_steps = model(images)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_steps += num_steps.mean().item()
    
    accuracy = 100 * correct / total
    avg_steps = total_steps / len(dataloader)
    return accuracy, avg_steps


def final_test_evaluation(model_paths, test_loader, device, test_dataset=None):
    """
    Perform final evaluation on test set using best models from each fold.
    Runs comprehensive analysis from analysis.py instead of basic metrics.
    """
    from collections import defaultdict
    from analysis import ModelAnalyzer
    
    print(f"\n{'='*60}")
    print("FINAL TEST SET EVALUATION")
    print(f"{'='*60}")
    
    # Determine number of classes from test dataset
    if hasattr(test_dataset, 'full_dataset'):
        num_classes = len(test_dataset.full_dataset.classes)
    elif hasattr(test_dataset, 'image_files'):
        num_classes = len(np.unique(test_dataset.image_files['label']))
    else:
        num_classes = 7
    
    all_predictions = defaultdict(list)
    all_labels = []
    fold_results = {}
    
    # Evaluate each fold's best model
    for fold_path in model_paths:
        if not os.path.exists(fold_path):
            print(f"Warning: Model file not found: {fold_path}")
            continue
            
        # Extract fold number from path (handles both wandb and checkpoint paths)
        fold_num = None
        if "best_model_fold_" in fold_path:
            fold_num = fold_path.split('best_model_fold_')[-1].split('.')[0]
        else:
            print(f"Warning: Cannot extract fold number from path: {fold_path}")
            continue
            
        print(f"\nEvaluating fold {fold_num} model from: {fold_path}")
        
        # Load model
        fold_model = RecursiveFER(
            in_channels=3, 
            num_classes=num_classes, 
            hidden_dim=128, 
            max_steps=10
        ).to(device)
        
        try:
            fold_model.load_state_dict(torch.load(fold_path, map_location=device))
        except Exception as e:
            print(f"Error loading model {fold_path}: {e}")
            del fold_model
            continue
        
        fold_model.eval()
        
        # Collect predictions for this fold
        fold_predictions = []
        fold_labels = []
        total_steps = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits, num_steps = fold_model(images)
                _, predicted = torch.max(logits.data, 1)
                
                fold_predictions.extend(predicted.cpu().numpy())
                fold_labels.extend(labels.cpu().numpy())
                total_steps += num_steps.mean().item()
        
        # Calculate fold accuracy
        fold_correct = sum(1 for pred, true in zip(fold_predictions, fold_labels) if pred == true)
        fold_accuracy = 100 * fold_correct / len(fold_labels)
        fold_avg_steps = total_steps / len(test_loader)
        
        fold_results[fold_num] = {
            'accuracy': fold_accuracy,
            'predictions': fold_predictions,
            'avg_steps': fold_avg_steps
        }
        
        print(f"Fold {fold_num} Test Accuracy: {fold_accuracy:.4f}")
        print(f"Fold {fold_num} Avg Steps: {fold_avg_steps:.2f}")
        
        # Store predictions for ensemble
        all_predictions[fold_num] = fold_predictions
        
        # Store labels (only once)
        if not all_labels:
            all_labels = fold_labels
        
        # Clean up model to free memory
        del fold_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    if not all_predictions:
        print("No valid models found for evaluation.")
        return
    
    # Calculate individual fold statistics
    fold_accuracies = [result['accuracy'] for result in fold_results.values()]
    print(f"\nIndividual Fold Results:")
    for fold_num, result in fold_results.items():
        print(f"  Fold {fold_num}: {result['accuracy']:.4f}")
    print(f"Mean Fold Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    
    # Ensemble prediction (majority vote)
    if len(all_predictions) > 1:
        print(f"\nEnsemble Evaluation:")
        ensemble_predictions = []
        for i in range(len(all_labels)):
            # Get predictions from all folds for this sample
            sample_predictions = [preds[i] for preds in all_predictions.values()]
            # Majority vote
            ensemble_pred = max(set(sample_predictions), key=sample_predictions.count)
            ensemble_predictions.append(ensemble_pred)
        
        ensemble_correct = sum(1 for pred, true in zip(ensemble_predictions, all_labels) if pred == true)
        ensemble_accuracy = 100 * ensemble_correct / len(all_labels)
        print(f"Ensemble Test Accuracy: {ensemble_accuracy:.4f}")
    
    print(f"\n{'='*60}")
    print("RUNNING COMPREHENSIVE ANALYSIS")
    print(f"{'='*60}")
    
    # Run comprehensive analysis on best performing model
    best_fold_num = max(fold_results.keys(), key=lambda k: fold_results[k]['accuracy'])
    best_fold_path = None
    
    for model_path in model_paths:
        if f"best_model_fold_{best_fold_num}" in model_path:
            best_fold_path = model_path
            break
    
    if best_fold_path and os.path.exists(best_fold_path):
        print(f"\nRunning detailed analysis on best performing model (Fold {best_fold_num})...")
        print(f"Best model path: {best_fold_path}")
        
        try:
            # Initialize analyzer with the best model
            analyzer = ModelAnalyzer(best_fold_path, str(device))
            
            # Run analysis
            results_df = analyzer.run_inference(test_loader)
            
            # Generate all analysis plots and reports
            output_dir = f"./analysis_results_fold_{best_fold_num}"
            os.makedirs(output_dir, exist_ok=True)
            
            print("Generating confidence vs steps plot...")
            analyzer.plot_confidence_vs_steps(os.path.join(output_dir, "confidence_vs_steps.png"))
            
            print("Generating steps per class plot...")
            analyzer.plot_steps_per_class(os.path.join(output_dir, "steps_per_class.png"))
            
            print("Generating extreme examples visualization...")
            analyzer.visualize_extreme_examples(test_loader, output_dir)
            
            print("Generating summary report...")
            report = analyzer.generate_summary_report(os.path.join(output_dir, "summary_report.txt"))
            
            print(f"\nComprehensive analysis complete! Results saved to {output_dir}")
            
        except Exception as e:
            print(f"Error during comprehensive analysis: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Warning: Could not find best model file for detailed analysis")
    
    print(f"{'='*60}")
    
    return fold_results


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
        import glob
        
        # First try to find models in wandb directories
        wandb_base_dir = "/kaggle/working/VRM/wandb"
        if os.path.exists(wandb_base_dir):
            # Look for run-* directories (Kaggle format: run-YYYYMMDD-hash)
            wandb_run_dirs = glob.glob(os.path.join(wandb_base_dir, "run-*"))
            for run_dir in wandb_run_dirs:
                # Check both the run directory and its "files" subdirectory
                for search_dir in [run_dir, os.path.join(run_dir, "files")]:
                    if os.path.exists(search_dir):
                        model_files = glob.glob(os.path.join(search_dir, "best_model_fold_*.pth"))
                        model_paths.extend(model_files)
        
        # If no models found in wandb, fallback to checkpoints directory
        if not model_paths:
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
    Train RecursiveFERModel model with ImageFolder dataset using GroupKFold cross-validation.
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
            print(f"  Class weights: {cv_loaders[args.single_fold]['class_weights']}")
            print(f"Num classes: {len(cv_loaders[args.single_fold]['class_weights'])}")
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RecursiveFERModel model with ImageFolder dataset and GroupKFold cross-validation.")
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