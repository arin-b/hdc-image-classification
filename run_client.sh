#!/bin/bash

# Flotilla HDC Client Runner
# This script runs HDC federated learning clients on Raspberry Pi

# Configuration - UPDATE THESE VALUES
NETWORK_NAME="flotilla-net"  # Replace with your network name from sheet
MQTT_IP="10.0.9.100"         # Replace with your MQTT IP from sheet
MQTT_PORT="1883"             # Replace with your MQTT port from sheet
MEMORY_LIMIT="512m"          # Adjust based on your Pi: 2048m for Pi4b 8GB, 512m for Pi4b 2GB, 100m for Pi3b
DATASET_ID="hdc_custom"      # Dataset identifier

# Classes for HDC
CLASSES=("bottle" "mouse" "mobile" "sharpner")

# Run containers based on available slots
# For 4 containers: (0 3), for 2 containers: (0 1)
for i in {0..3}; do
    CLASS_NAME=${CLASSES[$i]}
    
    echo "Starting HDC client container $i for class $CLASS_NAME"
    
    docker run -d \
        --name hdc_client_${i} \
        --network $NETWORK_NAME \
        --memory $MEMORY_LIMIT \
        --cpuset-cpus $i \
        -e CLIENT_ID=$i \
        -e CLASS_NAME=$CLASS_NAME \
        -e MQTT_HOST=$MQTT_IP \
        -e MQTT_PORT=$MQTT_PORT \
        -e DATASET_ID=$DATASET_ID \
        -v /home/wschool/hdc_data/${CLASS_NAME}:/app/data/${CLASS_NAME} \
        flotilla-client:latest \
        python federated_train.py
        
    echo "Started container hdc_client_${i} for class $CLASS_NAME"
done

echo "All HDC client containers started successfully!"
echo "Check container status with: docker ps"