# HDC Federated Learning - Reading Sequence Guide

## 📚 Complete Understanding Path (Start to End)

### Phase 1: Core HDC Concepts (Foundation)
**Read in this exact order:**

1. **`README.md`** (Project root)
   - Overview of HDC vs traditional ML
   - Key principles: "Learning by addition, not optimization"
   - Architecture overview

2. **`hdc/config.py`**
   - Core parameters (HD_DIM, INPUT_DIM, CLASS_MAP)
   - Random seed for reproducibility
   - Image processing settings

3. **`hdc/projection.py`**
   - Random bipolar projection matrix generation
   - Feature → hypervector encoding
   - Core HDC mathematical operations

4. **`hdc/features.py`**
   - CNN feature extractor (frozen MobileNetV2)
   - Image preprocessing pipeline
   - Feature dimension reduction

5. **`hdc/classifier.py`**
   - Centroid-based classification
   - Learning by vector accumulation
   - Cosine similarity inference

### Phase 2: Original Implementation (Single Machine)
**Understand the baseline:**

6. **`hdc/train.py`**
   - End-to-end training pipeline
   - Dataset loading and processing
   - Single-machine HDC training

7. **`hdc/compression.py`** (Optional)
   - Holographic Reduced Representations
   - Centroid compression for bandwidth efficiency
   - HRR binding operations

### Phase 3: Federated Architecture (Distribution)
**Core federated components:**

8. **`federated_train.py`**
   - MQTT-enabled federated client
   - Single-class training per container
   - Client-server communication protocol

9. **`federated_aggregator.py`**
   - Centralized aggregation server
   - Global model coordination
   - Inference demonstration

### Phase 4: Flotilla Integration (Production)
**Production deployment:**

10. **`flo_server.py`**
    - Flotilla server integration
    - Client registration and coordination
    - MQTT message handling

11. **`flo_session.py`**
    - Session management and monitoring
    - Federated learning workflow control
    - Performance tracking

12. **`config/hdc_config.yml`**
    - Flotilla-specific configuration
    - MQTT broker settings
    - Client deployment parameters

### Phase 5: Deployment Infrastructure (Operations)
**Docker and deployment:**

13. **`Dockerfile`**
    - Container definition
    - Dependency management
    - Multi-architecture support

14. **`docker-compose.yml`**
    - Local testing environment
    - Service orchestration
    - Network configuration

15. **`run_client.sh`**
    - Raspberry Pi deployment script
    - Container resource allocation
    - Flotilla client configuration

### Phase 6: Testing and Validation (Verification)
**Testing approaches:**

16. **`test_local.sh`**
    - Automated local testing
    - System verification
    - Pre-deployment validation

17. **`LAPTOP_TRAINING.md`**
    - Local development guide
    - Multiple training methods
    - Quick testing procedures

### Phase 7: Production Deployment (Operations)
**Deployment guides:**

18. **`DEPLOYMENT.md`**
    - Complete Raspberry Pi deployment
    - Flotilla integration steps
    - Troubleshooting guide

19. **`prepare_data.sh`**
    - Data preparation for Pi deployment
    - Directory structure setup
    - Permission configuration

### Phase 8: Project Overview (Summary)
**Final understanding:**

20. **`PROJECT_SUMMARY.md`**
    - Complete system architecture
    - Key features and benefits
    - Performance expectations

## 🎯 Quick Understanding Path (Essential Only)

If you need to understand quickly, read these 8 files in order:

1. **`README.md`** - What is HDC?
2. **`hdc/config.py`** - Key parameters
3. **`hdc/projection.py`** - Core HDC math
4. **`hdc/train.py`** - Basic training
5. **`federated_train.py`** - Federated client
6. **`federated_aggregator.py`** - Federated server
7. **`docker-compose.yml`** - How to run
8. **`LAPTOP_TRAINING.md`** - How to test

## 🔍 Deep Dive Path (Research/Development)

For complete technical understanding:

### Core Algorithm (Deep)
- `hdc/projection.py` - Mathematical foundations
- `hdc/classifier.py` - Learning algorithm
- `hdc/compression.py` - Advanced techniques

### Federated System (Deep)
- `federated_train.py` - Client implementation
- `federated_aggregator.py` - Server implementation
- `flo_server.py` - Production coordination

### Infrastructure (Deep)
- `Dockerfile` - Container architecture
- `run_client.sh` - Deployment automation
- `config/hdc_config.yml` - System configuration

## 📋 Practical Implementation Path

To actually run and modify the system:

### Step 1: Understand Core (30 min)
1. `README.md`
2. `hdc/config.py`
3. `hdc/train.py`

### Step 2: Test Locally (15 min)
4. `LAPTOP_TRAINING.md`
5. Run: `docker-compose up`

### Step 3: Understand Federated (45 min)
6. `federated_train.py`
7. `federated_aggregator.py`
8. `docker-compose.yml`

### Step 4: Deploy to Production (60 min)
9. `DEPLOYMENT.md`
10. `run_client.sh`
11. `flo_server.py`

## 🧠 Conceptual Understanding Order

### HDC Theory
1. `README.md` - High-level concepts
2. `hdc/projection.py` - Mathematical basis
3. `hdc/classifier.py` - Learning mechanism

### System Architecture
4. `federated_train.py` - Client design
5. `federated_aggregator.py` - Server design
6. `flo_server.py` - Production coordination

### Deployment Strategy
7. `Dockerfile` - Containerization
8. `run_client.sh` - Edge deployment
9. `DEPLOYMENT.md` - Complete workflow

## ⚡ Minimum Viable Understanding

To just run the system successfully:

1. **`LAPTOP_TRAINING.md`** - How to run locally
2. **`docker-compose.yml`** - What services are needed
3. **`hdc/config.py`** - What parameters to change
4. **`DEPLOYMENT.md`** - How to deploy to Pi

## 🔧 Modification/Extension Path

To modify or extend the system:

### Core Algorithm Changes
1. `hdc/config.py` - Parameters
2. `hdc/projection.py` - Encoding logic
3. `hdc/classifier.py` - Learning/inference

### Federated System Changes
4. `federated_train.py` - Client behavior
5. `federated_aggregator.py` - Server logic
6. `config/hdc_config.yml` - System settings

### Deployment Changes
7. `Dockerfile` - Container setup
8. `run_client.sh` - Pi deployment
9. `docker-compose.yml` - Local testing

## 📊 File Importance Ranking

### Critical (Must Read)
- `README.md` - Project foundation
- `hdc/config.py` - Core parameters
- `federated_train.py` - Client implementation
- `LAPTOP_TRAINING.md` - How to run

### Important (Should Read)
- `hdc/projection.py` - HDC algorithm
- `federated_aggregator.py` - Server implementation
- `docker-compose.yml` - System orchestration
- `DEPLOYMENT.md` - Production deployment

### Useful (Nice to Read)
- `hdc/train.py` - Original implementation
- `flo_server.py` - Flotilla integration
- `PROJECT_SUMMARY.md` - Complete overview

### Optional (Reference)
- `hdc/compression.py` - Advanced features
- `prepare_data.sh` - Data setup
- `test_local.sh` - Testing automation

## 🎯 Success Checkpoints

After each phase, you should understand:

**Phase 1**: What HDC is and how it differs from neural networks
**Phase 2**: How single-machine HDC training works
**Phase 3**: How federated HDC distributes learning
**Phase 4**: How Flotilla manages production deployment
**Phase 5**: How Docker containers package the system
**Phase 6**: How to test and validate the system
**Phase 7**: How to deploy to Raspberry Pi devices
**Phase 8**: Complete system architecture and capabilities

Follow this sequence for complete project mastery! 🚀