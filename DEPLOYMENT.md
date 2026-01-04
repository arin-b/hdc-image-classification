# HDC Federated Learning Deployment Guide

## Overview
This guide provides complete steps to deploy the HDC (Hyperdimensional Computing) image classification system using Docker containers with Flotilla for federated learning across Raspberry Pi devices.

## Architecture
- **4 Client Containers**: One per class (bottle, mouse, mobile, sharpner)
- **1 Aggregator Container**: Collects and combines centroids
- **1 MQTT Broker**: Handles communication between containers
- **Flotilla Integration**: Manages distributed deployment

## Prerequisites
- Raspberry Pi devices with Docker installed
- Access to Flotilla server and MQTT broker
- Dataset images organized by class
- Network connectivity between devices

## Step-by-Step Deployment

### 1. Server Setup (Flotilla Server Container)

```bash
# SSH into the Flotilla server container
ssh root@<Server Container IP> -p <Port>
# Password: root123

# Navigate to fedml directory
cd /fedml-ng && export PATH=/opt/conda/bin:$PATH

# Copy HDC project files to server
# (Upload the entire hdc-image-classification-main directory)

# Start the Flotilla server
python flo_server.py
```

### 2. Raspberry Pi Setup

#### 2.1 Initial Pi Configuration
```bash
# SSH into each Raspberry Pi
ssh wschool@<DEVICE IP>
# Password: wschool

# Load the Flotilla client image
docker load -i flotilla-client-arm.tar.gz

# Verify image loaded
docker image ls
```

#### 2.2 Join Docker Swarm
```bash
# For teams T1-T8:
docker swarm join --token SWMTKN-1-3uy7wsk54037m7rzt6toxmdwrbpmb8upwe643mtn98gynqtv91-celnjoqb6ize6buo3mgnm8h91 10.24.24.32:2377

# For teams T9-T16:
docker swarm join --token SWMTKN-1-3x45gzklzr94wooe0hvznf2itccrm1wucezs22yk95v42lwf9m-f1wxes3ajjxcrwnpbrp5yhar4 10.24.24.31:2377
```

#### 2.3 Enable Memory Control
```bash
sudo nano /boot/cmdline.txt
# Add to end of line: cgroup_enable=cpuset cgroup_enable=memory cgroup_memory=1
# Reboot: sudo reboot
```

#### 2.4 Prepare Data Directories
```bash
# Run data preparation script
bash prepare_data.sh

# Copy your dataset images to:
# /home/wschool/hdc_data/bottle/
# /home/wschool/hdc_data/mouse/
# /home/wschool/hdc_data/mobile/
# /home/wschool/hdc_data/sharpner/
```

### 3. Configure Flotilla Client Script

Edit `run_client.sh` with your specific values:

```bash
vim run_client.sh

# Update these values:
NETWORK_NAME="flotilla-net"        # From your team sheet
MQTT_IP="10.0.9.100"              # From your team sheet  
MQTT_PORT="1883"                  # From your team sheet
MEMORY_LIMIT="512m"               # Based on your Pi model
DATASET_ID="hdc_custom"           # Dataset identifier
```

Memory recommendations:
- Pi4b 8GB: `2048m`
- Pi4b 2GB: `512m`  
- Pi3b: `100m`

### 4. Launch Client Containers

```bash
# Make script executable
chmod +x run_client.sh

# Launch all client containers
bash run_client.sh

# Verify containers are running
docker ps
```

### 5. Start Federated Learning Session

In the second terminal on the Flotilla server:

```bash
# Start the federated learning session
python flo_session.py config/hdc_config.yml --federated_server_endpoint localhost:12345
```

## Expected Workflow

1. **Client Registration**: Each container registers with the aggregator
2. **Training Phase**: Clients train on their local class data
3. **Centroid Sharing**: Clients send their learned centroids to aggregator
4. **Global Aggregation**: Aggregator combines all centroids into global model
5. **Inference Demo**: System demonstrates classification on test images

## Monitoring and Debugging

### Check Container Status
```bash
# List running containers
docker ps

# Check container logs
docker logs hdc_client_0
docker logs hdc_client_1
docker logs hdc_client_2
docker logs hdc_client_3
```

### MQTT Message Monitoring
```bash
# Install MQTT client tools
sudo apt-get install mosquitto-clients

# Monitor all HDC messages
mosquitto_sub -h <MQTT_IP> -p <MQTT_PORT> -t "hdc/#"
```

### Resource Monitoring
```bash
# Check memory usage
docker stats

# Check system resources
free -h
htop
```

## Configuration Files

### Key Configuration Parameters

**config/hdc_config.yml**:
- `num_clients: 4` - Number of federated clients
- `hd_dimension: 4096` - Hypervector dimension
- `feature_dimension: 128` - CNN feature dimension
- `mqtt.host` - MQTT broker IP address

**hdc/config.py**:
- `CLASS_MAP` - Mapping of class names to IDs
- `HD_DIM` - Hypervector dimension
- `INPUT_DIM` - Feature vector dimension

## Troubleshooting

### Common Issues

1. **Container Memory Issues**
   - Reduce `MEMORY_LIMIT` in run_client.sh
   - Check available memory: `free -h`

2. **MQTT Connection Failed**
   - Verify MQTT_IP and MQTT_PORT in configuration
   - Check network connectivity: `ping <MQTT_IP>`

3. **Data Loading Errors**
   - Ensure data directories exist and contain images
   - Check file permissions: `ls -la /home/wschool/hdc_data/`

4. **Docker Swarm Issues**
   - Verify swarm membership: `docker node ls`
   - Rejoin swarm if needed

### Performance Optimization

1. **Memory Allocation**
   - Adjust memory limits based on Pi model
   - Monitor usage with `docker stats`

2. **CPU Pinning**
   - Containers are pinned to specific CPU cores
   - Modify `--cpuset-cpus` in run_client.sh if needed

3. **Network Optimization**
   - Use overlay network for container communication
   - Monitor network latency between devices

## Expected Results

- **Training Time**: ~2-5 minutes per client
- **Aggregation Time**: ~30 seconds
- **Inference Latency**: <100ms per image
- **Memory Usage**: 100-512MB per container
- **Network Traffic**: Minimal (only centroids transmitted)

## File Structure

```
hdc-image-classification-main/
├── hdc/                    # Core HDC implementation
├── federated_train.py      # Federated client
├── federated_aggregator.py # Federated aggregator
├── flo_server.py          # Flotilla server
├── flo_session.py         # Flotilla session manager
├── config/                # Configuration files
├── Dockerfile             # Container definition
├── docker-compose.yml     # Local testing
├── run_client.sh          # Pi deployment script
└── DEPLOYMENT.md          # This file
```

## Success Indicators

✅ All 4 client containers start successfully
✅ Clients register with aggregator
✅ Training completes on all clients
✅ Centroids are successfully aggregated
✅ Global model performs inference
✅ Classification accuracy > 75%

## Support

For issues or questions:
1. Check container logs first
2. Verify network connectivity
3. Ensure data is properly formatted
4. Monitor MQTT message flow
5. Check resource utilization