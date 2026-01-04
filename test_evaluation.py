"""
test_evaluation.py

Quick test script to verify train.py and evaluate_model.py work correctly.
"""

import os
import sys

def test_basic_training():
    """Test basic training without evaluation."""
    print("=== Testing Basic Training ===")
    
    # Change to hdc directory
    os.chdir('hdc')
    
    # Import and run basic training
    from train import train
    from config import CLASS_MAP
    
    print(f"Classes: {list(CLASS_MAP.keys())}")
    
    # Train on dataset
    data_dir = r"D:\hdc-image-classification-main\hdc_dtaset"
    if not os.path.exists(data_dir):
        print(f"Dataset not found at {data_dir}")
        return False
        
    classifier = train(data_dir)
    print(f"Training complete! Centroids shape: {classifier.centroids.shape}")
    
    # Quick inference test
    import torch
    from features import FeatureExtractor, get_image_transform
    from projection import generate_random_projection, encode_hypervector
    from config import INPUT_DIM, HD_DIM, RANDOM_SEED
    from PIL import Image
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = FeatureExtractor(INPUT_DIM).to(device).eval()
    projection_matrix = generate_random_projection(INPUT_DIM, HD_DIM, RANDOM_SEED)
    transform = get_image_transform()
    
    # Test on first image of first class
    first_class = list(CLASS_MAP.keys())[0]
    class_dir = os.path.join(data_dir, first_class)
    img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if img_files:
        img_path = os.path.join(class_dir, img_files[0])
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = feature_extractor(x).cpu().numpy().squeeze()
        
        hv = encode_hypervector(features, projection_matrix)
        pred = classifier.predict(hv)
        
        print(f"Test inference: {first_class} -> predicted class {pred}")
        
    return True

def test_evaluation():
    """Test evaluation functionality."""
    print("\n=== Testing Evaluation ===")
    
    try:
        # Go back to root directory
        os.chdir('..')
        
        # Test evaluate_model.py
        from evaluate_model import quick_evaluate
        
        data_dir = "hdc_dtaset"
        if not os.path.exists(data_dir):
            print(f"Dataset not found at {data_dir}")
            return False
            
        metrics = quick_evaluate(data_dir)
        print(f"Evaluation complete! Accuracy: {metrics['accuracy']*100:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"Evaluation test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing HDC Training and Evaluation System")
    print("=" * 50)
    
    # Test basic training
    success1 = test_basic_training()
    
    # Test evaluation
    success2 = test_evaluation()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        
    print("Basic training:", "✅" if success1 else "❌")
    print("Evaluation:", "✅" if success2 else "❌")