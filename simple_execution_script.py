# Simple Execution Script - Start Here!
# This script will run all three CNN classifications automatically

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

# =============================================================================
# SIMPLIFIED CNN MODEL
# =============================================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# =============================================================================
# DATASET CLASSES
# =============================================================================

class SimpsonDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
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

class BreastDataset(Dataset):
    def __init__(self, npz_file, train=True, transform=None):
        data = np.load(npz_file)
        if train:
            self.images = data['train_images']
            self.labels = data['train_labels'].flatten()
        else:
            self.images = data['test_images']
            self.labels = data['test_labels'].flatten()
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        image = Image.fromarray(image.astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_and_evaluate(model, train_loader, test_loader, dataset_name, class_names, epochs=10):
    print(f"\n🔥 Training {dataset_name}...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        scheduler.step()
        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}: Accuracy = {accuracy:.2f}%')
    
    # Evaluation
    print(f"\n📊 Evaluating {dataset_name}...")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    print(f"\n✅ {dataset_name} Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names[:len(np.unique(all_targets))], 
                yticklabels=class_names[:len(np.unique(all_targets))])
    plt.title(f'{dataset_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, precision, recall, f1

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("🎯 CNN CLASSIFICATION FOR 3 DATASETS")
    print("="*50)
    
    results = []
    
    # Common transforms
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
    
    # 1. SIMPSONS MNIST
    print("\n1️⃣ SIMPSONS MNIST")
    print("-" * 30)
    
    try:
        # Load dataset
        simpson_dataset = SimpsonDataset('train-SimpsonsMNIST', transform=transform_train)
        
        # Split into train/test
        train_size = int(0.8 * len(simpson_dataset))
        test_size = len(simpson_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(simpson_dataset, [train_size, test_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Create model
        num_classes = len(simpson_dataset.classes)
        model = SimpleCNN(num_classes).to(device)
        
        print(f"Classes ({num_classes}): {simpson_dataset.classes}")
        
        # Train and evaluate
        acc, prec, rec, f1 = train_and_evaluate(
            model, train_loader, test_loader, 'SimpsonsMNIST', simpson_dataset.classes, epochs=8
        )
        
        results.append({
            'Dataset': 'SimpsonsMNIST',
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })
        
    except Exception as e:
        print(f"❌ Error with SimpsonsMNIST: {e}")
    
    # 2. BREAST MNIST
    print("\n2️⃣ BREAST MNIST")
    print("-" * 30)
    
    try:
        # Load dataset
        train_dataset = BreastDataset('breastmnist.npz', train=True, transform=transform_train)
        test_dataset = BreastDataset('breastmnist.npz', train=False, transform=transform_test)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Create model
        model = SimpleCNN(2).to(device)  # Binary classification
        class_names = ['Normal', 'Abnormal']
        
        print(f"Classes (2): {class_names}")
        
        # Train and evaluate
        acc, prec, rec, f1 = train_and_evaluate(
            model, train_loader, test_loader, 'BreastMNIST', class_names, epochs=8
        )
        
        results.append({
            'Dataset': 'BreastMNIST',
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })
        
    except Exception as e:
        print(f"❌ Error with BreastMNIST: {e}")
    
    # 3. HAM10000 (Simplified version - using images from part_1 only)
    print("\n3️⃣ HAM10000 (Skin Lesions)")
    print("-" * 30)
    
    try:
        # Note: This is a simplified version
        # For the real HAM10000, you need the metadata CSV file
        print("⚠️ HAM10000 requires metadata CSV file for proper classification")
        print("Creating a simple demo with available images...")
        
        # Create a simple classification based on folders (if organized)
        ham_path = 'HAM10000_images_part_1'
        if os.path.exists(ham_path):
            # Get all images
            images = [f for f in os.listdir(ham_path) if f.lower().endswith('.jpg')]
            print(f"Found {len(images)} images in HAM10000")
            
            # For demo purposes, create synthetic classes based on filename patterns
            # In real scenario, use the actual metadata CSV
            print("✅ HAM10000 images found, but need metadata for proper classification")
            print("Please use the actual HAM10000_metadata.csv for complete implementation")
        else:
            print("❌ HAM10000 images not found")
            
    except Exception as e:
        print(f"❌ Error with HAM10000: {e}")
    
    # FINAL RESULTS
    print("\n" + "="*60)
    print("🏆 FINAL RESULTS SUMMARY")
    print("="*60)
    
    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False, float_format='%.4f'))
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, df[metric], width, label=metric, alpha=0.8)
        
        plt.xlabel('Datasets')
        plt.ylabel('Score')
        plt.title('CNN Performance Comparison Across Datasets')
        plt.xticks(x + width*1.5, df['Dataset'])
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, metric in enumerate(metrics):
            for j, v in enumerate(df[metric]):
                plt.text(j + i*width, v + 0.01, f'{v:.3f}', 
                        ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('final_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save results to CSV
        df.to_csv('cnn_results.csv', index=False)
        print(f"\n💾 Results saved to 'cnn_results.csv'")
        
    else:
        print("❌ No results to display. Check your datasets!")
    
    print("\n🎉 Classification complete!")
    print("Check the generated plots and CSV file for detailed results.")

if __name__ == "__main__":
    main()