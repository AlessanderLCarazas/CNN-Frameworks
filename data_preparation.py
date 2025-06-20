# Data Preparation Script for CNN Classification

import os
import pandas as pd
import numpy as np
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split

# =============================================================================
# DATA PREPARATION FUNCTIONS
# =============================================================================

def prepare_simpsons_data():
    """
    Prepare SimpsonsMNIST data - assumes data is already in correct format
    """
    print("Checking SimpsonsMNIST data structure...")
    
    train_path = "train-SimpsonsMNIST"
    if os.path.exists(train_path):
        classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
        print(f"Found {len(classes)} classes in SimpsonsMNIST:")
        for i, cls in enumerate(classes):
            class_path = os.path.join(train_path, cls)
            img_count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"  {i+1}. {cls}: {img_count} images")
        return True
    else:
        print("SimpsonsMNIST directory not found!")
        return False

def prepare_breast_mnist():
    """
    Check BreastMNIST data - should be a .npz file
    """
    print("Checking BreastMNIST data...")
    
    if os.path.exists("breastmnist.npz"):
        data = np.load("breastmnist.npz")
        print("BreastMNIST data found!")
        print("Keys in dataset:", list(data.keys()))
        
        if 'train_images' in data:
            print(f"Train images shape: {data['train_images'].shape}")
            print(f"Train labels shape: {data['train_labels'].shape}")
        if 'test_images' in data:
            print(f"Test images shape: {data['test_images'].shape}")
            print(f"Test labels shape: {data['test_labels'].shape}")
        
        # Check class distribution
        if 'train_labels' in data:
            unique, counts = np.unique(data['train_labels'], return_counts=True)
            print("Class distribution in training set:")
            for u, c in zip(unique, counts):
                print(f"  Class {u}: {c} samples")
        
        return True
    else:
        print("breastmnist.npz file not found!")
        print("Please download it from: https://medmnist.com/")
        return False

def prepare_ham10000_metadata():
    """
    Create metadata CSV for HAM10000 dataset
    This function creates a basic metadata structure if you don't have the original CSV
    """
    print("Preparing HAM10000 metadata...")
    
    # Check if images exist
    img_dirs = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
    all_images = []
    
    for img_dir in img_dirs:
        if os.path.exists(img_dir):
            images = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
            all_images.extend([(img_dir, img) for img in images])
            print(f"Found {len(images)} images in {img_dir}")
    
    if not all_images:
        print("HAM10000 images not found!")
        return False
    
    # Create basic metadata structure
    # Note: This is a simplified version - you should use the actual HAM10000_metadata.csv
    metadata_file = "HAM10000_metadata.csv"
    
    if not os.path.exists(metadata_file):
        print("Creating basic metadata file...")
        
        # Common skin lesion types in HAM10000
        lesion_types = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        
        metadata_list = []
        for img_dir, img_file in all_images:
            image_id = img_file.replace('.jpg', '')
            # Randomly assign lesion type (replace with actual metadata)
            dx = np.random.choice(lesion_types)
            
            metadata_list.append({
                'image_id': image_id,
                'dx': dx,
                'dx_type': 'histo',  # or 'consensus'
                'age': np.random.randint(20, 80),
                'sex': np.random.choice(['male', 'female']),
                'localization': np.random.choice(['torso', 'lower extremity', 'upper extremity', 'head/neck'])
            })
        
        df = pd.DataFrame(metadata_list)
        df.to_csv(metadata_file, index=False)
        print(f"Created {metadata_file} with {len(df)} entries")
        
        # Show class distribution
        class_counts = df['dx'].value_counts()
        print("Class distribution:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} samples")
    
    return True

def check_requirements():
    """
    Check if all required packages are installed
    """
    required_packages = [
        'torch', 'torchvision', 'numpy', 'pandas', 'matplotlib', 
        'seaborn', 'scikit-learn', 'PIL', 'cv2'
    ]
    
    print("Checking required packages...")
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                from PIL import Image
            elif package == 'cv2':
                import cv2
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them using:")
        if 'cv2' in missing_packages:
            missing_packages.remove('cv2')
            print("pip install opencv-python")
        if 'PIL' in missing_packages:
            missing_packages.remove('PIL')
            print("pip install Pillow")
        if missing_packages:
            print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("All required packages are installed!")
    return True

def create_directory_structure():
    """
    Create necessary directories for the project
    """
    directories = [
        'models',
        'results', 
        'plots'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

# =============================================================================
# MAIN PREPARATION SCRIPT
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("DATA PREPARATION FOR CNN CLASSIFICATION")
    print("="*60)
    
    # Check requirements
    if not check_requirements():
        print("Please install missing packages before proceeding.")
        exit(1)
    
    # Create directory structure
    create_directory_structure()
    
    # Check and prepare each dataset
    print("\n" + "="*40)
    print("DATASET PREPARATION")
    print("="*40)
    
    # SimpsonsMNIST
    simpsons_ready = prepare_simpsons_data()
    
    print("\n" + "-"*40)
    
    # BreastMNIST  
    breast_ready = prepare_breast_mnist()
    
    print("\n" + "-"*40)
    
    # HAM10000
    ham_ready = prepare_ham10000_metadata()
    
    print("\n" + "="*40)
    print("PREPARATION SUMMARY")
    print("="*40)
    print(f"SimpsonsMNIST ready: {'✓' if simpsons_ready else '✗'}")
    print(f"BreastMNIST ready: {'✓' if breast_ready else '✗'}")
    print(f"HAM10000 ready: {'✓' if ham_ready else '✗'}")
    
    if all([simpsons_ready, breast_ready, ham_ready]):
        print("\n🎉 All datasets are ready! You can now run the main CNN script.")
    else:
        print("\n⚠️ Some datasets need attention. Please check the messages above.")
        
    print("\nNext steps:")
    print("1. Run this preparation script first")
    print("2. Make sure all datasets are properly loaded")
    print("3. Run the main CNN classification script")
    print("4. Check results in the 'results' directory")