#!/bin/bash

# Data preparation script for HDC federated learning
# This script prepares data for distribution to Raspberry Pi containers

echo "=== HDC Federated Learning Data Preparation ==="

# Create data directories on Raspberry Pi
echo "Creating data directories..."

# Classes in the dataset
CLASSES=("bottle" "mouse" "mobile" "sharpner")

# Create base directory structure
mkdir -p /home/wschool/hdc_data
mkdir -p /home/wschool/hdc_test_data

for class_name in "${CLASSES[@]}"; do
    echo "Setting up data for class: $class_name"
    
    # Create training data directory
    mkdir -p /home/wschool/hdc_data/$class_name
    
    # Create test data directory  
    mkdir -p /home/wschool/hdc_test_data/$class_name
    
    echo "Created directories for $class_name"
done

echo "Data directory structure created successfully!"
echo ""
echo "Next steps:"
echo "1. Copy your dataset images to the appropriate directories:"
echo "   - Training data: /home/wschool/hdc_data/<class_name>/"
echo "   - Test data: /home/wschool/hdc_test_data/<class_name>/"
echo ""
echo "2. Ensure images are in supported formats: .jpg, .jpeg, .png"
echo ""
echo "3. Run the Flotilla client containers using: bash run_client.sh"

# Display directory structure
echo ""
echo "Created directory structure:"
tree /home/wschool/hdc_data 2>/dev/null || find /home/wschool/hdc_data -type d
tree /home/wschool/hdc_test_data 2>/dev/null || find /home/wschool/hdc_test_data -type d