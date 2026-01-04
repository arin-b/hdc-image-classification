"""
augment_data.py

Data augmentation script to generate 500 images per class.
"""

import os
import random
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

def augment_image(image):
    """Apply random augmentations to an image."""
    
    # Random rotation (-15 to 15 degrees)
    angle = random.uniform(-15, 15)
    image = image.rotate(angle, fillcolor=(255, 255, 255))
    
    # Random brightness (0.8 to 1.2)
    brightness = random.uniform(0.8, 1.2)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    
    # Random contrast (0.8 to 1.2)
    contrast = random.uniform(0.8, 1.2)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    
    # Random horizontal flip (50% chance)
    if random.random() > 0.5:
        image = ImageOps.mirror(image)
    
    # Random crop and resize
    width, height = image.size
    crop_size = random.uniform(0.8, 1.0)
    new_width = int(width * crop_size)
    new_height = int(height * crop_size)
    
    left = random.randint(0, width - new_width)
    top = random.randint(0, height - new_height)
    
    image = image.crop((left, top, left + new_width, top + new_height))
    image = image.resize((224, 224))
    
    return image

def augment_dataset(input_dir, output_dir, target_count=500):
    """Augment dataset to target_count images per class."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name in ["bottle", "mouse", "mobile", "sharpner"]:
        class_input_dir = os.path.join(input_dir, class_name)
        class_output_dir = os.path.join(output_dir, class_name)
        
        if not os.path.exists(class_input_dir):
            print(f"Warning: {class_input_dir} not found")
            continue
            
        os.makedirs(class_output_dir, exist_ok=True)
        
        # Get original images
        original_images = [f for f in os.listdir(class_input_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Class {class_name}: {len(original_images)} original images")
        
        # Copy original images first
        for i, img_file in enumerate(original_images):
            img_path = os.path.join(class_input_dir, img_file)
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224, 224))
            img.save(os.path.join(class_output_dir, f"orig_{i:03d}.jpg"))
        
        # Generate augmented images
        generated = len(original_images)
        while generated < target_count:
            # Pick random original image
            orig_img_file = random.choice(original_images)
            orig_img_path = os.path.join(class_input_dir, orig_img_file)
            
            try:
                img = Image.open(orig_img_path).convert("RGB")
                aug_img = augment_image(img)
                
                aug_filename = f"aug_{generated:03d}.jpg"
                aug_img.save(os.path.join(class_output_dir, aug_filename))
                
                generated += 1
                
                if generated % 50 == 0:
                    print(f"  Generated {generated}/{target_count} images for {class_name}")
                    
            except Exception as e:
                print(f"Error processing {orig_img_path}: {e}")
                continue
        
        print(f"✅ Class {class_name}: {generated} total images")

if __name__ == "__main__":
    input_dir = r"D:\hdc-image-classification-main\hdc_dtaset"
    output_dir = r"D:\hdc-image-classification-main\hdc_augmented"
    
    print("Starting data augmentation...")
    augment_dataset(input_dir, output_dir, target_count=500)
    print("Data augmentation complete!")