# HDC Federated Learning with Flotilla - Complete Implementation

## Project Overview

This implementation provides a complete Docker-based federated learning system for Hyperdimensional Computing (HDC) image classification using Flotilla for distributed deployment across Raspberry Pi devices.

## Key Features

✅ **Federated HDC Learning**: Each container trains on one class, centroids are aggregated
✅ **Flotilla Integration**: Full compatibility with Flotilla server and MQTT communication  
✅ **Docker Containerization**: Isolated, reproducible environments
✅ **Raspberry Pi Optimized**: Memory and CPU optimizations for edge devices
✅ **Real-time Communication**: MQTT-based coordination between containers
✅ **No Backpropagation**: Pure HDC learning through vector accumulation
✅ **Edge-Friendly**: Low memory, low latency, bandwidth efficient

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raspberry Pi  │    │   Raspberry Pi  │    │   Raspberry Pi  │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │HDC Client 0 │ │    │ │HDC Client 1 │ │    │ │HDC Client 2 │ │
│ │(bottle)     │ │    │ │(mouse)      │ │    │ │(mobile)     │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   MQTT Broker   │
                    │   (Flotilla)    │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Aggregator    │
                    │   (Server)      │
                    └─────────────────┘
```

## File Structure

```
hdc-image-classification-main/
├── hdc/                          # Original HDC implementation
│   ├── config.py                 # Updated with 4 classes
│   ├── features.py               # CNN feature extractor
│   ├── projection.py             # HDC projection logic
│   ├── classifier.py             # HDC classifier
│   ├── compression.py            # Optional compression
│   ├── train.py                  # Original training script
│   └── validate.py               # Original validation (not used)
├── federated_train.py            # 🆕 Federated client
├── federated_aggregator.py       # 🆕 Federated aggregator
├── flo_server.py                 # 🆕 Flotilla server integration
├── flo_session.py                # 🆕 Flotilla session manager
├── config/
│   └── hdc_config.yml            # 🆕 Flotilla configuration
├── Dockerfile                    # 🆕 Container definition
├── docker-compose.yml            # 🆕 Local testing
├── mosquitto.conf                # 🆕 MQTT configuration
├── run_client.sh                 # 🆕 Pi deployment script
├── prepare_data.sh               # 🆕 Data preparation
├── test_local.sh                 # 🆕 Local testing
├── DEPLOYMENT.md                 # 🆕 Deployment guide
├── requirements.txt              # Updated with MQTT/YAML
└── hdc_dtaset/                   # Your 4-class dataset
    ├── bottle/
    ├── mouse/
    ├── mobile/
    └── sharpner/
```

## Quick Start

### 1. Local Testing
```bash
# Test the system locally first
chmod +x test_local.sh
./test_local.sh

# Monitor the federated learning process
docker-compose logs -f aggregator
```

### 2. Raspberry Pi Deployment

#### Server Setup
```bash
# SSH into Flotilla server container
ssh root@<Server Container IP> -p <Port>

# Start Flotilla server
cd /fedml-ng && export PATH=/opt/conda/bin:$PATH
python flo_server.py
```

#### Client Setup (on each Pi)
```bash
# SSH into Raspberry Pi
ssh wschool@<DEVICE IP>

# Load Flotilla client image
docker load -i flotilla-client-arm.tar.gz

# Join Docker swarm (use your team's token)
docker swarm join --token <YOUR_TOKEN> <SERVER_IP>:2377

# Prepare data directories
bash prepare_data.sh

# Configure and run clients
vim run_client.sh  # Update MQTT_IP, NETWORK_NAME, etc.
bash run_client.sh
```

#### Start Federated Session
```bash
# In second terminal on Flotilla server
python flo_session.py config/hdc_config.yml --federated_server_endpoint localhost:12345
```

## Key Components Explained

### 1. Federated Client (`federated_train.py`)
- Trains HDC classifier on single class data
- Communicates via MQTT with aggregator
- Sends learned centroids for global aggregation
- Handles inference requests

### 2. Federated Aggregator (`federated_aggregator.py`)
- Coordinates training across all clients
- Collects and combines centroids from all classes
- Performs global inference using combined model
- Manages federated learning workflow

### 3. Flotilla Integration (`flo_server.py`, `flo_session.py`)
- `flo_server.py`: Manages client coordination and MQTT communication
- `flo_session.py`: Controls federated learning sessions and monitoring

### 4. Docker Configuration
- **Dockerfile**: Optimized for both x86 and ARM architectures
- **docker-compose.yml**: Local testing with all components
- **run_client.sh**: Production deployment script for Pi

## Configuration

### Key Parameters

**HDC Configuration** (`hdc/config.py`):
```python
CLASS_MAP = {
    "bottle": 0,
    "mouse": 1, 
    "mobile": 2,
    "sharpner": 3,
}
HD_DIM = 4096          # Hypervector dimension
INPUT_DIM = 128        # CNN feature dimension
```

**Flotilla Configuration** (`config/hdc_config.yml`):
```yaml
num_clients: 4         # One per class
hd_dimension: 4096     # Must match HD_DIM
mqtt:
  host: "10.0.9.100"   # Your MQTT broker IP
  port: 1883
```

**Deployment Configuration** (`run_client.sh`):
```bash
NETWORK_NAME="flotilla-net"    # From your team sheet
MQTT_IP="10.0.9.100"          # From your team sheet
MEMORY_LIMIT="512m"           # Based on Pi model
```

## Expected Performance

- **Training Time**: 2-5 minutes per client
- **Memory Usage**: 100-512MB per container
- **Network Traffic**: Minimal (only centroids transmitted)
- **Inference Latency**: <100ms per image
- **Expected Accuracy**: >75% on 4-class dataset

## Workflow

1. **Initialization**: Clients register with aggregator via MQTT
2. **Training**: Each client trains HDC classifier on its class data
3. **Centroid Sharing**: Clients send learned centroids to aggregator
4. **Aggregation**: Aggregator combines centroids into global model
5. **Inference**: Global model classifies test images
6. **Results**: System reports accuracy and performance metrics

## Advantages of This Approach

### HDC Benefits
- **No Backpropagation**: Learning through simple vector addition
- **Single Pass Training**: No epochs or iterations needed
- **Memory Efficient**: Constant memory usage regardless of dataset size
- **Fast Inference**: Simple cosine similarity computation

### Federated Benefits
- **Privacy Preserving**: Raw data never leaves local containers
- **Bandwidth Efficient**: Only centroids (not gradients) transmitted
- **Fault Tolerant**: System continues if some clients fail
- **Scalable**: Easy to add more classes/clients

### Edge Computing Benefits
- **Low Resource Usage**: Optimized for Raspberry Pi constraints
- **Real-time Capable**: Fast training and inference
- **Offline Capable**: Clients can train without constant connectivity
- **Deployment Ready**: Production-ready containerized system

## Monitoring and Debugging

### Container Monitoring
```bash
# Check all containers
docker ps

# Monitor specific container
docker logs -f hdc_client_0

# Check resource usage
docker stats
```

### MQTT Monitoring
```bash
# Monitor all HDC messages
mosquitto_sub -h <MQTT_IP> -t "hdc/#"

# Monitor specific topics
mosquitto_sub -h <MQTT_IP> -t "hdc/training_complete"
```

### System Health
```bash
# Check Pi resources
free -h
htop

# Check network connectivity
ping <MQTT_IP>
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Container OOM | Reduce `MEMORY_LIMIT` in run_client.sh |
| MQTT connection failed | Verify `MQTT_IP` and network connectivity |
| Data loading errors | Check data directory structure and permissions |
| Training timeout | Increase `round_duration` in config |
| Low accuracy | Verify dataset quality and class balance |

## Success Criteria

✅ All 4 client containers start successfully  
✅ Clients register with aggregator via MQTT  
✅ Training completes on all clients  
✅ Centroids successfully aggregated  
✅ Global model performs inference  
✅ Classification accuracy >75%  
✅ System runs within memory constraints  
✅ End-to-end latency <5 minutes  

## Next Steps

1. **Test Locally**: Run `./test_local.sh` to verify system
2. **Deploy to Pi**: Follow DEPLOYMENT.md instructions
3. **Monitor Performance**: Use provided monitoring tools
4. **Scale Up**: Add more classes or clients as needed
5. **Optimize**: Tune hyperparameters for your specific dataset

This implementation provides a complete, production-ready federated HDC learning system optimized for edge deployment with Flotilla integration.