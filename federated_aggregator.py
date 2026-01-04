"""
federated_aggregator.py

Federated HDC aggregator for Flotilla integration.
Collects centroids from all clients and performs global inference.
"""

import json
import time
import numpy as np
import torch
from PIL import Image
import paho.mqtt.client as mqtt
from threading import Event
import uuid

from hdc.config import *
from hdc.features import FeatureExtractor, get_image_transform
from hdc.projection import generate_random_projection, encode_hypervector
from hdc.classifier import HDCClassifier

class FederatedHDCAggregator:
    def __init__(self, mqtt_host, mqtt_port=1883):
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        
        # MQTT setup
        self.mqtt_client = mqtt.Client(client_id="hdc_aggregator")
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        
        # Global model components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = FeatureExtractor(INPUT_DIM)
        self.feature_extractor.to(self.device).eval()
        self.projection_matrix = generate_random_projection(INPUT_DIM, HD_DIM, RANDOM_SEED)
        
        # Global classifier
        self.global_classifier = HDCClassifier(len(CLASS_MAP), HD_DIM)
        self.transform = get_image_transform()
        
        # Client management
        self.expected_clients = len(CLASS_MAP)
        self.received_centroids = {}
        self.training_complete_clients = set()
        
        # Synchronization events
        self.all_trained = Event()
        self.aggregation_complete = Event()
        
    def on_connect(self, client, userdata, flags, rc):
        print(f"Aggregator connected with result code {rc}")
        client.subscribe("hdc/training_complete")
        client.subscribe("hdc/centroid_update")
        client.subscribe("hdc/inference_response")
        
    def on_message(self, client, userdata, msg):
        topic = msg.topic
        data = json.loads(msg.payload.decode())
        
        if topic == "hdc/training_complete":
            client_id = data["client_id"]
            class_name = data["class_name"]
            self.training_complete_clients.add(client_id)
            print(f"Training complete for client {client_id} (class {class_name})")
            
            if len(self.training_complete_clients) == self.expected_clients:
                print("All clients completed training. Starting aggregation...")
                self.start_aggregation()
                
        elif topic == "hdc/centroid_update":
            client_id = data["client_id"]
            class_id = data["class_id"]
            centroid = np.array(data["centroid"])
            
            self.received_centroids[class_id] = centroid
            print(f"Received centroid from client {client_id} for class {data['class_name']}")
            
            if len(self.received_centroids) == self.expected_clients:
                self.aggregate_centroids()
                
        elif topic == "hdc/inference_response":
            # Handle inference responses from clients
            print(f"Inference response from client {data['client_id']}: "
                  f"class {data['class_name']}, similarity {data['similarity']:.4f}")
    
    def start_aggregation(self):
        """Request centroids from all clients"""
        self.mqtt_client.publish("hdc/aggregate", "start")
        
    def aggregate_centroids(self):
        """Aggregate received centroids into global model"""
        print("Aggregating centroids...")
        
        for class_id, centroid in self.received_centroids.items():
            self.global_classifier.centroids[class_id] = centroid
            
        print("Global model aggregation complete!")
        self.aggregation_complete.set()
        
        # Start inference demo
        self.demo_inference()
    
    def demo_inference(self):
        """Demonstrate inference on test images"""
        print("\n=== Starting Inference Demo ===")
        
        # Test with sample images from each class
        test_results = []
        
        for class_name in CLASS_MAP.keys():
            test_dir = f"/app/test_data/{class_name}"
            if not os.path.exists(test_dir):
                print(f"Test directory {test_dir} not found, skipping...")
                continue
                
            # Test first image in directory
            test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not test_files:
                continue
                
            img_path = os.path.join(test_dir, test_files[0])
            result = self.infer_image(img_path, class_name)
            test_results.append(result)
            
        # Print results
        print("\n=== Inference Results ===")
        for result in test_results:
            print(f"Image: {result['true_class']} -> Predicted: {result['predicted_class']} "
                  f"(confidence: {result['confidence']:.4f})")
            
        accuracy = sum(1 for r in test_results if r['correct']) / len(test_results) if test_results else 0
        print(f"\nAccuracy: {accuracy:.2%}")
    
    def infer_image(self, img_path, true_class):
        """Perform inference on a single image"""
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.feature_extractor(x).cpu().numpy().squeeze()
        
        hv = encode_hypervector(features, self.projection_matrix)
        predicted_class_id = self.global_classifier.predict(hv)
        
        # Get class name from ID
        class_id_to_name = {v: k for k, v in CLASS_MAP.items()}
        predicted_class = class_id_to_name[predicted_class_id]
        
        # Calculate confidence (max similarity)
        hv_norm = np.linalg.norm(hv) + 1e-8
        centroid_norms = np.linalg.norm(self.global_classifier.centroids, axis=1) + 1e-8
        similarities = (self.global_classifier.centroids @ hv) / (centroid_norms * hv_norm)
        confidence = np.max(similarities)
        
        return {
            'true_class': true_class,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'correct': true_class == predicted_class
        }
    
    def start_training(self):
        """Initiate training across all clients"""
        print("Starting federated training...")
        self.mqtt_client.publish("hdc/start_training", "start")
        
    def run(self):
        """Main aggregator loop"""
        self.mqtt_client.connect(self.mqtt_host, self.mqtt_port, 60)
        print("Aggregator is running...")
        
        # Start training after a short delay
        time.sleep(5)
        self.start_training()
        
        self.mqtt_client.loop_forever()

if __name__ == "__main__":
    mqtt_host = os.environ.get("MQTT_HOST", "localhost")
    mqtt_port = int(os.environ.get("MQTT_PORT", "1883"))
    
    aggregator = FederatedHDCAggregator(mqtt_host, mqtt_port)
    aggregator.run()