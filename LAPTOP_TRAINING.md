# HDC Federated Learning - Laptop Training Guide

## Prerequisites

- Windows/macOS/Linux laptop with Docker installed
- Python 3.9+ (optional, for direct Python execution)
- At least 4GB RAM available
- Dataset in `hdc_dtaset/` directory

## Method 1: Docker-Based Training (Recommended)

### Step 1: Verify Setup
```bash
# Check Docker is running
docker --version
docker info

# Verify dataset structure
ls hdc_dtaset/
# Should show: bottle/ mouse/ mobile/ sharpner/
```

### Step 2: Build and Run
```bash
# Navigate to project directory
cd hdc-image-classification-main

# Build Docker image
docker build -t hdc-federated:latest .

# Start complete federated system
docker-compose up
```

### Step 3: Monitor Training
```bash
# In new terminal - watch aggregator logs
docker-compose logs -f aggregator

# Watch all containers
docker-compose logs -f

# Check container status
docker-compose ps
```

### Step 4: View Results
Training will complete automatically. Look for:
- "Training complete for client X"
- "Global model aggregation complete!"
- "Accuracy: XX.XX%"

### Step 5: Cleanup
```bash
# Stop all containers
docker-compose down

# Remove containers and images (optional)
docker-compose down --rmi all
```

## Method 2: Direct Python Execution

### Step 1: Install Dependencies
```bash
# Create virtual environment
python -m venv hdc_env
source hdc_env/bin/activate  # Linux/macOS
# OR
hdc_env\Scripts\activate     # Windows

# Install requirements
pip install -r requirements.txt
```

### Step 2: Run Original Training
```bash
# Simple single-machine training
cd hdc
python train.py

# This trains on all classes together (not federated)
```

### Step 3: Run Validation
```bash
# Test on CIFAR-10 (downloads automatically)
python validate.py
```

## Method 3: Simulated Federated Training

### Step 1: Start MQTT Broker
```bash
# Terminal 1 - Start MQTT broker
docker run -it -p 1883:1883 eclipse-mosquitto:2.0
```

### Step 2: Start Aggregator
```bash
# Terminal 2 - Start aggregator
python federated_aggregator.py
```

### Step 3: Start Clients (4 terminals)
```bash
# Terminal 3 - Client 0 (bottle)
CLIENT_ID=0 CLASS_NAME=bottle MQTT_HOST=localhost python federated_train.py

# Terminal 4 - Client 1 (mouse)  
CLIENT_ID=1 CLASS_NAME=mouse MQTT_HOST=localhost python federated_train.py

# Terminal 5 - Client 2 (mobile)
CLIENT_ID=2 CLASS_NAME=mobile MQTT_HOST=localhost python federated_train.py

# Terminal 6 - Client 3 (sharpner)
CLIENT_ID=3 CLASS_NAME=sharpner MQTT_HOST=localhost python federated_train.py
```

## Quick Test Script

### Step 1: Create Test Script
```bash
# Create quick_test.py
cat > quick_test.py << 'EOF'
import os
import sys
sys.path.append('hdc')

from hdc.config import *
from hdc.train import train

print("=== HDC Quick Test ===")
print(f"Classes: {list(CLASS_MAP.keys())}")
print(f"HD Dimension: {HD_DIM}")

# Train on dataset
print("Starting training...")
classifier = train("hdc_dtaset")

print("Training complete!")
print(f"Centroids shape: {classifier.centroids.shape}")

# Test inference on first image of each class
from PIL import Image
from hdc.features import FeatureExtractor, get_image_transform
from hdc.projection import generate_random_projection, encode_hypervector
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = FeatureExtractor(INPUT_DIM).to(device).eval()
transform = get_image_transform()
R = generate_random_projection(INPUT_DIM, HD_DIM, RANDOM_SEED)

correct = 0
total = 0

for class_name, class_id in CLASS_MAP.items():
    class_dir = f"hdc_dtaset/{class_name}"
    if os.path.exists(class_dir):
        img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if img_files:
            img_path = os.path.join(class_dir, img_files[0])
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = feature_extractor(x).cpu().numpy().squeeze()
            
            hv = encode_hypervector(features, R)
            pred = classifier.predict(hv)
            
            print(f"Class {class_name}: predicted {pred}, actual {class_id} {'✓' if pred == class_id else '✗'}")
            
            if pred == class_id:
                correct += 1
            total += 1

if total > 0:
    accuracy = (correct / total) * 100
    print(f"\nAccuracy: {accuracy:.1f}% ({correct}/{total})")
else:
    print("No test images found!")
EOF
```

### Step 2: Run Test
```bash
python quick_test.py
```

## Expected Output

### Docker Method
```
hdc_mqtt        | Starting MQTT broker...
hdc_aggregator  | Aggregator is running...
hdc_client_bottle | Client 0 for class bottle is running...
hdc_client_mouse  | Client 1 for class mouse is running...
hdc_client_mobile | Client 2 for class mobile is running...
hdc_client_sharpner | Client 3 for class sharpner is running...

# Training messages
hdc_aggregator  | All clients completed training. Starting aggregation...
hdc_aggregator  | Global model aggregation complete!
hdc_aggregator  | === Inference Results ===
hdc_aggregator  | Accuracy: 85.00%
```

### Python Method
```
=== HDC Quick Test ===
Classes: ['bottle', 'mouse', 'mobile', 'sharpner']
HD Dimension: 4096
Starting training...
Training complete!
Centroids shape: (4, 4096)
Class bottle: predicted 0, actual 0 ✓
Class mouse: predicted 1, actual 1 ✓
Class mobile: predicted 2, actual 2 ✓
Class sharpner: predicted 3, actual 3 ✓

Accuracy: 100.0% (4/4)
```

## Troubleshooting

### Docker Issues
```bash
# If containers fail to start
docker-compose down
docker system prune -f
docker-compose up --build

# Check logs for specific container
docker-compose logs aggregator
```

### Python Issues
```bash
# If import errors
pip install --upgrade -r requirements.txt

# If CUDA errors
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Data Issues
```bash
# Verify dataset structure
find hdc_dtaset -name "*.jpg" | head -10

# Check permissions
chmod -R 755 hdc_dtaset/
```

## Performance Expectations

- **Training Time**: 1-3 minutes total
- **Memory Usage**: ~2GB for all containers
- **Accuracy**: 75-90% on your 4-class dataset
- **CPU Usage**: Moderate during training, low during inference

## Success Indicators

✅ All containers start without errors
✅ Training completes on all classes  
✅ Centroids are successfully aggregated
✅ Inference produces reasonable accuracy
✅ No memory or resource issues

## Next Steps After Laptop Testing

1. **Verify Results**: Ensure accuracy >75%
2. **Check Logs**: Review training process
3. **Test Inference**: Try with new images
4. **Deploy to Pi**: Use DEPLOYMENT.md guide
5. **Scale Up**: Add more classes if needed