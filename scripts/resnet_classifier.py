import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import numpy as np
from tqdm import tqdm
import json
import random

from dataset import GOESDataset

class CycloneClassifier(nn.Module):
    """Lightweight ResNet-18 classifier for tropical cyclone detection."""
    
    def __init__(self, num_classes=2, pretrained=True, use_single_channel=False):
        """
        Args:
            num_classes (int): Number of output classes (2 for binary: cyclone/no-cyclone)
            pretrained (bool): Whether to use pretrained ImageNet weights
            use_single_channel (bool): If True, modify conv1 for 1-channel input (no pretrained weights)
                                       If False, expect 3-channel RGB input (can use pretrained weights)
        """
        super(CycloneClassifier, self).__init__()
        
        self.use_single_channel = use_single_channel
        
        # Load ResNet-18 (lightweight option)
        self.resnet = models.resnet18(pretrained=pretrained)
        
        if use_single_channel:
            # Modify first conv layer to accept 1-channel input (grayscale satellite imagery)
            # Note: This discards pretrained weights for the first layer
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Otherwise, keep 3-channel input to use pretrained weights
        
        # Replace final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        # If single channel, input shape is [B, 1, H, W]
        # If RGB, input shape is [B, 3, H, W]
        return self.resnet(x)


class CycloneTrainer:
    """Training pipeline for cyclone classifier."""
    
    def __init__(self, model, train_loader, val_loader, device, learning_rate=1e-3, log_file='training_log.csv', tensorboard_dir='runs'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.log_file = log_file
        
        # TensorBoard writer
        self.writer = SummaryWriter(tensorboard_dir)
        print(f"TensorBoard logging to: {tensorboard_dir}")
        print(f"Start TensorBoard with: tensorboard --logdir={tensorboard_dir}")
        
        # Label mapping
        self.label_map = {'negative': 0, 'positive': 1}
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write('epoch,train_loss,train_acc,val_loss,val_acc,learning_rate\n')
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            images = batch['patch'].to(self.device)
            # Convert string labels to integers
            labels = torch.tensor([self.label_map[cat] for cat in batch['category']]).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Log to TensorBoard (every batch)
            self.writer.add_scalar('Loss/train_batch', loss.item(), self.global_step)
            self.writer.add_scalar('Accuracy/train_batch', 100. * predicted.eq(labels).sum().item() / labels.size(0), self.global_step)
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch in pbar:
                images = batch['patch'].to(self.device)
                # Convert string labels to integers
                labels = torch.tensor([self.label_map[cat] for cat in batch['category']]).to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs, save_path='best_model.pth'):
        """Full training loop."""
        print(f"Training on device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Logging to: {self.log_file}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log epoch metrics to TensorBoard
            self.writer.add_scalar('Loss/train_epoch', train_loss, epoch)
            self.writer.add_scalar('Loss/val_epoch', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train_epoch', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val_epoch', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Log to CSV file
            with open(self.log_file, 'a') as f:
                f.write(f'{epoch+1},{train_loss:.6f},{train_acc:.4f},{val_loss:.6f},{val_acc:.4f},{current_lr:.8f}\n')
            
            # Print epoch summary
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.8f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, save_path)
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Close TensorBoard writer
        self.writer.close()
        
        print("\n" + "="*50)
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }


def split_dataset(dataset, val_split=0.2, seed=42):
    """
    Split dataset into train and validation sets.
    
    Args:
        dataset: PyTorch Dataset object
        val_split (float): Fraction of data to use for validation
        seed (int): Random seed for reproducibility
    
    Returns:
        train_dataset, val_dataset: Subset objects
    """
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get dataset size
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    # Shuffle indices
    random.shuffle(indices)
    
    # Calculate split point
    split_idx = int(dataset_size * (1 - val_split))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"Dataset split (seed={seed}):")
    print(f"  Total samples: {dataset_size}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def get_device():
    """Get the best available device (MPS for Mac, CUDA for others, CPU fallback)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# Example usage
if __name__ == "__main__":
    # Import your GOESDataset here
    # from your_dataset_module import GOESDataset
    
    # Configuration
    METADATA_PATH = '/Users/dylanwhite/Documents/Projects/tropical-cv/data/training/image_data.json'
    VAL_SPLIT = 0.2  # 20% for validation
    SEED = 42  # For reproducibility
    BATCH_SIZE = 16  # Adjust based on your Mac's memory
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-3
    PATCH_SIZE = 512
    NUM_WORKERS = 2  # For DataLoader
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create full dataset
    full_dataset = GOESDataset(
        image_metadata_path=METADATA_PATH,
        patch_size=PATCH_SIZE,
        augment=True,
        center_bias=0.6,
        three_channel=True
    )
    
    # Split into train and validation
    train_dataset, val_dataset = split_dataset(
        full_dataset,
        val_split=VAL_SPLIT,
        seed=SEED
    )
    
    # Note: For validation, we want to disable augmentation
    # Since we're using Subset, we need to handle this differently
    # The augmentation will still be applied, but you could modify GOESDataset
    # to accept an index list and handle augmentation per-sample if needed
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )
    
    # Create model
    # Option 1: RGB with pretrained weights (RECOMMENDED for limited data)
    model = CycloneClassifier(num_classes=2, pretrained=True, use_single_channel=False)
    
    # Option 2: Single channel, no pretrained weights
    # model = CycloneClassifier(num_classes=2, pretrained=False, use_single_channel=True)
    
    # Create trainer
    trainer = CycloneTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=LEARNING_RATE,
        log_file='training_log.csv'
    )
    
    # Train
    history = trainer.train(num_epochs=NUM_EPOCHS, save_path='cyclone_classifier.pth')
    
    print("\nTraining history saved to 'training_history.json'")
    print("Training log saved to 'training_log.csv'")
    print("Best model saved to 'cyclone_classifier.pth'")