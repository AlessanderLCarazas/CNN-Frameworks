# CNN Classification for SimpsonsMNIST, BreastMNIST, and HAM10000
# Complete solution with all required metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import os
from PIL import Image
import cv2
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =============================================================================
# 1. CUSTOM DATASET CLASSES
# =============================================================================

class SimpsonsMNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(class_path, img_name), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class BreastMNISTDataset(Dataset):
    def __init__(self, npz_file, transform=None, train=True):
        data = np.load(npz_file)
        if train:
            self.images = data['train_images']
            self.labels = data['train_labels']
        else:
            self.images = data['test_images']
            self.labels = data['test_labels']
        
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx][0]  # Remove extra dimension
        
        # Convert to PIL Image
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)  # Convert grayscale to RGB
        image = Image.fromarray(image.astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class HAM10000Dataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.metadata = pd.read_csv(csv_file)
        
        # Create class mapping
        self.classes = sorted(self.metadata['dx'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for _, row in self.metadata.iterrows():
            img_path = os.path.join(root_dir, f"{row['image_id']}.jpg")
            if os.path.exists(img_path):
                self.samples.append((img_path, self.class_to_idx[row['dx']]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# =============================================================================
# 2. CNN ARCHITECTURES
# =============================================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, input_size=(64, 64)):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Calculate the size after conv layers
        with torch.no_grad():
            x = torch.zeros(1, 3, input_size[0], input_size[1])
            x = self.features(x)
            flat_size = x.view(1, -1).size(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flat_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# =============================================================================
# 3. TRAINING FUNCTIONS
# =============================================================================

def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        scheduler.step()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)
    
    return model, train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, test_loader, class_names):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(all_targets, all_predictions, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return accuracy, precision, recall, f1

# =============================================================================
# 4. MAIN EXECUTION FOR EACH DATASET
# =============================================================================

def run_simpsons_mnist():
    print("="*60)
    print("SIMPSONS MNIST CLASSIFICATION")
    print("="*60)
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load datasets
    train_dataset = SimpsonsMNISTDataset('train-SimpsonsMNIST', transform=transform_train)
    
    # Split train into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Apply test transform to validation
    val_dataset.dataset.transform = transform_test
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Model
    num_classes = len(train_dataset.dataset.classes)
    model = SimpleCNN(num_classes=num_classes).to(device)
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {train_dataset.dataset.classes}")
    
    # Train model
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, num_epochs=15, lr=0.001
    )
    
    # Evaluate
    accuracy, precision, recall, f1 = evaluate_model(model, val_loader, train_dataset.dataset.classes)
    
    return {
        'dataset': 'SimpsonsMNIST',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def run_breast_mnist():
    print("="*60)
    print("BREAST MNIST CLASSIFICATION")
    print("="*60)
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load datasets
    train_dataset = BreastMNISTDataset('breastmnist.npz', transform=transform_train, train=True)
    test_dataset = BreastMNISTDataset('breastmnist.npz', transform=transform_test, train=False)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model
    num_classes = 2  # Binary classification
    model = SimpleCNN(num_classes=num_classes).to(device)
    
    print(f"Number of classes: {num_classes}")
    print("Classes: ['Normal', 'Abnormal']")
    
    # Train model
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, test_loader, num_epochs=15, lr=0.001
    )
    
    # Evaluate
    class_names = ['Normal', 'Abnormal']
    accuracy, precision, recall, f1 = evaluate_model(model, test_loader, class_names)
    
    return {
        'dataset': 'BreastMNIST',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def run_ham10000():
    print("="*60)
    print("HAM10000 CLASSIFICATION")
    print("="*60)
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load datasets (assuming you have HAM10000_metadata.csv)
    # You'll need to create this from your HAM10000 data
    dataset = HAM10000Dataset('HAM10000_images_part_1', 'HAM10000_metadata.csv', transform=transform_train)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model
    num_classes = len(dataset.classes)
    model = SimpleCNN(num_classes=num_classes, input_size=(128, 128)).to(device)
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {dataset.classes}")
    
    # Train model
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, num_epochs=20, lr=0.001
    )
    
    # Evaluate
    accuracy, precision, recall, f1 = evaluate_model(model, test_loader, dataset.classes)
    
    return {
        'dataset': 'HAM10000',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    results = []
    
    # Run SimpsonsMNIST
    try:
        simpsons_results = run_simpsons_mnist()
        results.append(simpsons_results)
    except Exception as e:
        print(f"Error running SimpsonsMNIST: {e}")
    
    # Run BreastMNIST
    try:
        breast_results = run_breast_mnist()
        results.append(breast_results)
    except Exception as e:
        print(f"Error running BreastMNIST: {e}")
    
    # Run HAM10000
    try:
        ham_results = run_ham10000()
        results.append(ham_results)
    except Exception as e:
        print(f"Error running HAM10000: {e}")
    
    # Print final results summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    # Create visualization
    if results:
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            datasets = [r['dataset'] for r in results]
            values = [r[metric] for r in results]
            
            axes[i].bar(datasets, values, color=['skyblue', 'lightcoral', 'lightgreen'][:len(datasets)])
            axes[i].set_title(f'{metric.title()} Comparison')
            axes[i].set_ylabel(metric.title())
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()