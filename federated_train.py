"""
federated_train.py

Federated HDC training client for Flotilla integration.
Each container trains on a single class and communicates centroids via MQTT.
"""

import os
import json
import time
import pickle
import numpy as np
import torch
from PIL import Image
import paho.mqtt.client as mqtt
from threading import Event

from hdc.config import *
from hdc.features import FeatureExtractor, get_image_transform
from hdc.projection import generate_random_projection, encode_hypervector
from hdc.classifier import HDCClassifier

class FederatedHDCClient:
    def __init__(self, client_id, class_name, mqtt_host, mqtt_port=1883):
        self.client_id = client_id
        self.class_name = class_name
        self.class_id = CLASS_MAP[class_name]
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        
        # MQTT setup
        self.mqtt_client = mqtt.Client(client_id=f"hdc_client_{client_id}")
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        
        # Training components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = FeatureExtractor(INPUT_DIM)
        self.feature_extractor.to(self.device).eval()
        self.projection_matrix = generate_random_projection(INPUT_DIM, HD_DIM, RANDOM_SEED)
        
        # Local classifier for this class only
        self.local_classifier = HDCClassifier(1, HD_DIM)  # Single class
        self.transform = get_image_transform()
        
        # Synchronization
        self.training_complete = Event()
        self.aggregation_complete = Event()
        
    def on_connect(self, client, userdata, flags, rc):
        print(f"Client {self.client_id} connected with result code {rc}")
        client.subscribe("hdc/start_training")
        client.subscribe("hdc/aggregate")
        client.subscribe("hdc/inference_request")
        
    def on_message(self, client, userdata, msg):
        topic = msg.topic
        
        if topic == "hdc/start_training":
            print(f"Client {self.client_id}: Starting training for class {self.class_name}")
            self.train_local()
            
        elif topic == "hdc/aggregate":
            print(f"Client {self.client_id}: Sending centroid for aggregation")
            self.send_centroid()
            
        elif topic == "hdc/inference_request":
            data = json.loads(msg.payload.decode())
            self.handle_inference_request(data)
    
    def train_local(self):
        """Train on local class data"""
        data_dir = f"/app/data/{self.class_name}"
        
        if not os.path.exists(data_dir):
            print(f"Warning: Data directory {data_dir} not found")
            return
            
        sample_count = 0
        for img_file in os.listdir(data_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(data_dir, img_file)
                img = Image.open(img_path).convert("RGB")
                
                x = self.transform(img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    features = self.feature_extractor(x).cpu().numpy().squeeze()
                
                hv = encode_hypervector(features, self.projection_matrix)
                self.local_classifier.add_sample(0, hv)  # Local class ID = 0
                sample_count += 1
        
        print(f"Client {self.client_id}: Trained on {sample_count} samples for class {self.class_name}")
        
        # Notify training completion
        self.mqtt_client.publish("hdc/training_complete", 
                                json.dumps({"client_id": self.client_id, "class_name": self.class_name}))
    
    def send_centroid(self):
        """Send local centroid to aggregator"""
        centroid_data = {
            "client_id": self.client_id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "centroid": self.local_classifier.centroids[0].tolist()
        }
        
        self.mqtt_client.publish("hdc/centroid_update", 
                                json.dumps(centroid_data))
        print(f"Client {self.client_id}: Sent centroid for class {self.class_name}")
    
    def handle_inference_request(self, request_data):
        """Handle inference request with feature vector"""
        features = np.array(request_data["features"])
        hv = encode_hypervector(features, self.projection_matrix)
        
        # Compute similarity with local centroid
        centroid = self.local_classifier.centroids[0]
        hv_norm = np.linalg.norm(hv) + 1e-8
        centroid_norm = np.linalg.norm(centroid) + 1e-8
        similarity = np.dot(centroid, hv) / (centroid_norm * hv_norm)
        
        response = {
            "client_id": self.client_id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "similarity": float(similarity),
            "request_id": request_data["request_id"]
        }
        
        self.mqtt_client.publish("hdc/inference_response", json.dumps(response))
    
    def run(self):
        """Main client loop"""
        self.mqtt_client.connect(self.mqtt_host, self.mqtt_port, 60)
        print(f"Client {self.client_id} for class {self.class_name} is running...")
        self.mqtt_client.loop_forever()

if __name__ == "__main__":
    client_id = int(os.environ.get("CLIENT_ID", "0"))
    class_name = os.environ.get("CLASS_NAME", "bottle")
    mqtt_host = os.environ.get("MQTT_HOST", "localhost")
    mqtt_port = int(os.environ.get("MQTT_PORT", "1883"))
    
    client = FederatedHDCClient(client_id, class_name, mqtt_host, mqtt_port)
    client.run()