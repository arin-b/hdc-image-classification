"""
split_data.py

Split augmented dataset into train/test sets.
"""

import os
import shutil
import random

def split_dataset(input_dir, output_dir, test_ratio=0.2):
    """Split dataset into train/test sets."""
    
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    for class_name in ["bottle", "mouse", "mobile", "sharpner"]:
        class_input_dir = os.path.join(input_dir, class_name)
        
        if not os.path.exists(class_input_dir):
            continue
            
        # Create class directories
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        # Get all images
        all_images = [f for f in os.listdir(class_input_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Shuffle and split
        random.shuffle(all_images)
        n_test = int(len(all_images) * test_ratio)
        
        test_images = all_images[:n_test]
        train_images = all_images[n_test:]
        
        # Copy test images
        for img_file in test_images:
            src = os.path.join(class_input_dir, img_file)
            dst = os.path.join(test_dir, class_name, img_file)
            shutil.copy2(src, dst)
        
        # Copy train images
        for img_file in train_images:
            src = os.path.join(class_input_dir, img_file)
            dst = os.path.join(train_dir, class_name, img_file)
            shutil.copy2(src, dst)
        
        print(f"Class {class_name}: {len(train_images)} train, {len(test_images)} test")

if __name__ == "__main__":
    input_dir = r"D:\hdc-image-classification-main\hdc_augmented"
    output_dir = r"D:\hdc-image-classification-main\hdc_split"
    
    random.seed(42)  # For reproducibility
    
    print("Splitting dataset...")
    split_dataset(input_dir, output_dir, test_ratio=0.2)
    print("Dataset split complete!")