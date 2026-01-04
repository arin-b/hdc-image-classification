#!/bin/bash

# Local testing script for HDC federated learning
# Run this before deploying to Raspberry Pi devices

echo "=== HDC Federated Learning Local Test ==="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "✅ Docker is running"

# Build the HDC image
echo "Building HDC Docker image..."
docker build -t hdc-federated:latest .

if [ $? -ne 0 ]; then
    echo "❌ Failed to build Docker image"
    exit 1
fi

echo "✅ Docker image built successfully"

# Check if dataset exists
if [ ! -d "hdc_dtaset" ]; then
    echo "❌ Dataset directory 'hdc_dtaset' not found"
    echo "Please ensure the dataset is in the correct location"
    exit 1
fi

echo "✅ Dataset directory found"

# Verify all classes are present
CLASSES=("bottle" "mouse" "mobile" "sharpner")
for class_name in "${CLASSES[@]}"; do
    if [ ! -d "hdc_dtaset/$class_name" ]; then
        echo "❌ Class directory 'hdc_dtaset/$class_name' not found"
        exit 1
    fi
    
    # Count images in directory
    image_count=$(find "hdc_dtaset/$class_name" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l)
    echo "✅ Class '$class_name': $image_count images"
done

# Start the federated learning system
echo ""
echo "Starting HDC federated learning system..."
echo "This will start:"
echo "  - 1 MQTT broker"
echo "  - 1 Aggregator"
echo "  - 4 Client containers (one per class)"

# Start with docker-compose
docker-compose up -d

if [ $? -ne 0 ]; then
    echo "❌ Failed to start containers"
    exit 1
fi

echo "✅ All containers started"

# Wait for containers to be ready
echo ""
echo "Waiting for containers to initialize..."
sleep 10

# Check container status
echo ""
echo "Container Status:"
docker-compose ps

# Monitor logs for a few seconds
echo ""
echo "Monitoring system startup (30 seconds)..."
echo "Press Ctrl+C to stop monitoring and continue"

timeout 30 docker-compose logs -f || true

echo ""
echo "=== Test Summary ==="
echo "✅ Docker image built successfully"
echo "✅ All containers started"
echo "✅ Dataset verified"

echo ""
echo "Next steps:"
echo "1. Check logs: docker-compose logs -f"
echo "2. Monitor training: docker-compose logs -f aggregator"
echo "3. Stop system: docker-compose down"
echo "4. Deploy to Raspberry Pi using DEPLOYMENT.md instructions"

echo ""
echo "Expected workflow:"
echo "1. Clients register with aggregator"
echo "2. Training starts automatically"
echo "3. Clients train on their class data"
echo "4. Centroids are aggregated"
echo "5. Global model performs inference"

echo ""
echo "🚀 Local test completed successfully!"