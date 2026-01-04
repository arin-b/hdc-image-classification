"""
flo_server.py

Flotilla server integration for HDC federated learning.
Manages client coordination and model aggregation.
"""

import json
import time
import logging
from threading import Thread
import paho.mqtt.client as mqtt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlotillaHDCServer:
    def __init__(self, mqtt_host="localhost", mqtt_port=1883):
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        
        # MQTT client for server coordination
        self.mqtt_client = mqtt.Client(client_id="flotilla_hdc_server")
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        
        # Client tracking
        self.registered_clients = set()
        self.training_status = {}
        self.aggregation_status = {}
        
        # Expected clients (4 classes)
        self.expected_clients = 4
        self.expected_classes = ["bottle", "mouse", "mobile", "sharpner"]
        
    def on_connect(self, client, userdata, flags, rc):
        logger.info(f"Flotilla HDC Server connected with result code {rc}")
        
        # Subscribe to all HDC topics
        topics = [
            "hdc/client_register",
            "hdc/training_complete", 
            "hdc/centroid_update",
            "hdc/inference_response"
        ]
        
        for topic in topics:
            client.subscribe(topic)
            logger.info(f"Subscribed to {topic}")
    
    def on_message(self, client, userdata, msg):
        topic = msg.topic
        
        try:
            data = json.loads(msg.payload.decode())
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received on topic {topic}")
            return
            
        if topic == "hdc/client_register":
            self.handle_client_registration(data)
            
        elif topic == "hdc/training_complete":
            self.handle_training_complete(data)
            
        elif topic == "hdc/centroid_update":
            self.handle_centroid_update(data)
            
        elif topic == "hdc/inference_response":
            self.handle_inference_response(data)
    
    def handle_client_registration(self, data):
        """Handle client registration"""
        client_id = data.get("client_id")
        class_name = data.get("class_name")
        
        self.registered_clients.add(client_id)
        logger.info(f"Client {client_id} registered for class {class_name}")
        
        if len(self.registered_clients) == self.expected_clients:
            logger.info("All clients registered. Starting federated training...")
            self.start_federated_training()
    
    def handle_training_complete(self, data):
        """Handle training completion from clients"""
        client_id = data.get("client_id")
        class_name = data.get("class_name")
        
        self.training_status[client_id] = {
            "class_name": class_name,
            "completed": True,
            "timestamp": time.time()
        }
        
        logger.info(f"Training completed for client {client_id} (class {class_name})")
        
        if len(self.training_status) == self.expected_clients:
            logger.info("All clients completed training. Starting aggregation...")
            self.start_aggregation()
    
    def handle_centroid_update(self, data):
        """Handle centroid updates from clients"""
        client_id = data.get("client_id")
        class_name = data.get("class_name")
        
        self.aggregation_status[client_id] = {
            "class_name": class_name,
            "centroid_received": True,
            "timestamp": time.time()
        }
        
        logger.info(f"Centroid received from client {client_id} (class {class_name})")
        
        if len(self.aggregation_status) == self.expected_clients:
            logger.info("All centroids received. Federated learning complete!")
            self.complete_federated_learning()
    
    def handle_inference_response(self, data):
        """Handle inference responses from clients"""
        client_id = data.get("client_id")
        class_name = data.get("class_name")
        similarity = data.get("similarity", 0.0)
        
        logger.info(f"Inference response from client {client_id} ({class_name}): {similarity:.4f}")
    
    def start_federated_training(self):
        """Initiate federated training across all clients"""
        logger.info("Broadcasting training start signal...")
        self.mqtt_client.publish("hdc/start_training", json.dumps({"command": "start"}))
    
    def start_aggregation(self):
        """Start centroid aggregation process"""
        logger.info("Broadcasting aggregation start signal...")
        self.mqtt_client.publish("hdc/aggregate", json.dumps({"command": "start"}))
    
    def complete_federated_learning(self):
        """Complete federated learning process"""
        logger.info("=== FEDERATED HDC LEARNING COMPLETED ===")
        
        # Log final status
        for client_id in self.registered_clients:
            training_info = self.training_status.get(client_id, {})
            aggregation_info = self.aggregation_status.get(client_id, {})
            
            logger.info(f"Client {client_id}: "
                       f"Class={training_info.get('class_name', 'unknown')}, "
                       f"Training={'✓' if training_info.get('completed') else '✗'}, "
                       f"Aggregation={'✓' if aggregation_info.get('centroid_received') else '✗'}")
        
        # Trigger inference demo
        self.demo_inference()
    
    def demo_inference(self):
        """Demonstrate inference capabilities"""
        logger.info("Starting inference demonstration...")
        
        # This would typically load test images and perform inference
        # For now, we'll just signal that inference is ready
        self.mqtt_client.publish("hdc/inference_ready", 
                                json.dumps({"status": "ready", "timestamp": time.time()}))
    
    def run(self):
        """Main server loop"""
        logger.info("Starting Flotilla HDC Server...")
        
        try:
            self.mqtt_client.connect(self.mqtt_host, self.mqtt_port, 60)
            logger.info(f"Connected to MQTT broker at {self.mqtt_host}:{self.mqtt_port}")
            
            # Start MQTT loop
            self.mqtt_client.loop_forever()
            
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise

if __name__ == "__main__":
    import os
    
    mqtt_host = os.environ.get("MQTT_HOST", "localhost")
    mqtt_port = int(os.environ.get("MQTT_PORT", "1883"))
    
    server = FlotillaHDCServer(mqtt_host, mqtt_port)
    server.run()