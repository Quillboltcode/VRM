
import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from dataset import RafDBDataset, ImageFolderDataset, get_default_transform, create_balanced_loaders, create_imagefolder_loaders
from loss import PonderLoss
from model import RecursiveFER

# Kaggle specific imports. These will fail if not in a Kaggle environment.
try:
    from kaggle_secrets import UserSecretsClient
except ImportError:
    UserSecretsClient = None

def train_single_fold(fold_data, args, device):
    """Train model on a single fold."""
    fold_num = fold_data['fold']
    train_loader = fold_data['train_loader']
    val_loader = fold_data['val_loader']
    class_weights = fold_data['class_weights']
    
    print(f"\n{'='*50}")
    print(f"Training Fold {fold_num + 1}")
    print(f"{'='*50}")
    
    # Model setup
    model = RecursiveFER(
        in_channels=3, 
        num_classes=7, 
        hidden_dim=args.hidden_dim, 
        max_steps=args.max_steps
    ).to(device)
    
    # Loss with class weights
    classification_loss = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    criterion = PonderLoss(classification_loss, lambda_ponder=args.lambda_ponder)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Wandb setup for this fold
    wandb.init(
        project="trm-fer-act", 
        config=args,
        name=f"fold_{fold_num}",
        reinit=True
    )
    wandb.watch(model)

    # Training loop
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

        # Validation
        val_acc, val_avg_steps = evaluate(model, val_loader, device)

        wandb.log({
            "epoch": epoch,
            "fold": fold_num,
            "train_loss": avg_loss,
            "train_class_loss": avg_class_loss,
            "train_ponder_cost": avg_ponder_cost,
            "train_avg_steps": avg_steps,
            "val_accuracy": val_acc,
            "val_avg_steps": val_avg_steps,
        })

        print(f"Fold {fold_num+1} Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}, Avg Steps: {val_avg_steps:.2f}")

        # Save best model for this fold
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_dir = wandb.run.dir if wandb.run else "."
            torch.save(model.state_dict(), os.path.join(output_dir, f"best_model_fold_{fold_num}.pth"))
        
        # Save checkpoint
        output_dir = wandb.run.dir if wandb.run else "."
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_fold_{fold_num}_epoch_{epoch+1}.pth"))

    wandb.finish()
    return best_val_acc


def train(args):
    # Wandb setup
    if UserSecretsClient is not None:
        user_secrets = UserSecretsClient()
        wandb_api_key = user_secrets.get_secret("wandb_api_secret")
        wandb.login(key=wandb_api_key)
    else:
        print("Wandb API key not found. Logging to console.")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data paths
    root_dir = args.root_dir
    if args.use_imagefolder:
        label_file = None
        print(f"Using ImageFolder dataset from: {root_dir}")
    else:
        label_file = os.path.join(root_dir, "EmoLabel/list_patition_label.txt")
        print(f"Using RAF-DB dataset from: {root_dir}")
    
    # Check if dataset exists and create dummy if needed
    if not args.use_imagefolder and label_file and not os.path.exists(label_file):
        print(f"Dataset not found at {label_file}. Creating dummy dataset for testing.")
        # Create dummy dataset structure
        os.makedirs(os.path.dirname(label_file), exist_ok=True)
        os.makedirs(os.path.join(root_dir, "Image/aligned"), exist_ok=True)
        
        # Create dummy label file
        with open(label_file, "w") as f:
            for i in range(100):
                f.write(f"train_{i+1:04d}_aligned.jpg {((i % 7) + 1)}\n")
                f.write(f"test_{i+1:04d}_aligned.jpg {((i % 7) + 1)}\n")
        
        # Create dummy images
        from PIL import Image
        for i in range(100):
            img = Image.new('RGB', (100, 100), color=(i*2, i*3, i*4))
            img.save(f"{root_dir}/Image/aligned/train_{i+1:04d}_aligned.jpg")
            img.save(f"{root_dir}/Image/aligned/test_{i+1:04d}_aligned.jpg")

    # Create balanced loaders with cross-validation
    print("Creating balanced data loaders with cross-validation...")
    try:
        if args.use_imagefolder:
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
        else:
            if label_file:
                cv_loaders = create_balanced_loaders(
                    root_dir=root_dir,
                    label_file=label_file,
                    batch_size=args.batch_size,
                    image_size=args.image_size,
                    num_workers=args.num_workers,
                    use_weighted_sampler=args.use_weighted_sampler,
                    n_splits=args.n_folds,
                    random_state=args.random_state
                )
            else:
                raise ValueError("label_file is required for RAF-DB dataset")
        
        if args.single_fold is not None:
            # Train on single specific fold
            if args.single_fold >= len(cv_loaders):
                print(f"Error: Fold {args.single_fold} not found. Available folds: 0-{len(cv_loaders)-1}")
                return
            print(f"Training on single fold {args.single_fold}...")
            fold_data = cv_loaders[args.single_fold]
            best_acc = train_single_fold(fold_data, args, device)
            print(f"\nFinal validation accuracy for fold {args.single_fold}: {best_acc:.4f}")
            
        elif args.use_cross_validation:
            print(f"Training with {args.n_folds}-fold cross-validation...")
            fold_accuracies = []
            
            for fold_data in cv_loaders:
                best_acc = train_single_fold(fold_data, args, device)
                fold_accuracies.append(best_acc)
            
            # Print cross-validation results
            print(f"\n{'='*50}")
            print("CROSS-VALIDATION RESULTS")
            print(f"{'='*50}")
            for i, acc in enumerate(fold_accuracies):
                print(f"Fold {i+1}: {acc:.4f}")
            print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
            print(f"{'='*50}")
            
        else:
            # Train on single split (first fold only)
            print("Training on single split...")
            fold_data = cv_loaders[0]
            best_acc = train_single_fold(fold_data, args, device)
            print(f"\nFinal validation accuracy: {best_acc:.4f}")
            
    except Exception as e:
        print(f"Error creating balanced loaders: {e}")
        print("Falling back to simple training setup...")
        
        # Fallback to original simple training
        train_transform = get_default_transform(image_size=args.image_size, is_train=True)
        test_transform = get_default_transform(image_size=args.image_size, is_train=False)

        if args.use_imagefolder:
            train_dataset = ImageFolderDataset(root_dir=root_dir, transform=train_transform, is_train=True, train_split=args.train_split)
            test_dataset = ImageFolderDataset(root_dir=root_dir, transform=test_transform, is_train=False, train_split=args.train_split)
        else:
            if label_file:
                train_dataset = RafDBDataset(root_dir=root_dir, label_file=label_file, transform=train_transform, is_train=True)
                test_dataset = RafDBDataset(root_dir=root_dir, label_file=label_file, transform=test_transform, is_train=False)
            else:
                raise ValueError("label_file is required for RAF-DB dataset")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        # Model, Loss, Optimizer
        model = RecursiveFER(in_channels=3, num_classes=7, hidden_dim=args.hidden_dim, max_steps=args.max_steps).to(device)
        classification_loss = torch.nn.CrossEntropyLoss()
        criterion = PonderLoss(classification_loss, lambda_ponder=args.lambda_ponder)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

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
            val_acc, val_avg_steps = evaluate(model, test_loader, device)
            print(f"Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}, Avg Steps: {val_avg_steps:.2f}")

            # Save checkpoint
            os.makedirs("./checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"./checkpoints/model_epoch_{epoch+1}.pth")

def evaluate(model, dataloader, device):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a RecursiveFER model.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension of the model")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of recursive steps")
    parser.add_argument("--lambda_ponder", type=float, default=0.01, help="Lambda for ponder cost")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--use_cross_validation", action="store_true", help="Use cross-validation instead of single split")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--single_fold", type=int, default=None, help="Train on a single specific fold (0-based)")
    parser.add_argument("--use_weighted_sampler", action="store_true", help="Use weighted sampler for class balancing")
    parser.add_argument("--use_imagefolder", action="store_true", help="Use ImageFolder dataset format instead of RAF-DB")
    parser.add_argument("--train_split", type=float, default=0.8, help="Train/validation split ratio for ImageFolder")
    parser.add_argument("--root_dir", type=str, default="/kaggle/input/rafdb", help="Root directory of the dataset")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    
    args = parser.parse_args()
    
    train(args)
