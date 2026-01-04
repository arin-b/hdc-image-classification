"""
flo_session.py

Flotilla session manager for HDC federated learning.
Coordinates the entire federated learning workflow.
"""

import json
import time
import argparse
import logging
from threading import Thread, Event
import paho.mqtt.client as mqtt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlotillaHDCSession:
    def __init__(self, config_file, server_endpoint="localhost:12345"):
        self.config_file = config_file
        self.server_endpoint = server_endpoint
        
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # MQTT setup
        mqtt_config = self.config.get("mqtt", {})
        self.mqtt_host = mqtt_config.get("host", "localhost")
        self.mqtt_port = mqtt_config.get("port", 1883)
        
        self.mqtt_client = mqtt.Client(client_id="flotilla_hdc_session")
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        
        # Session state
        self.session_active = False
        self.clients_ready = set()
        self.training_complete = Event()
        self.aggregation_complete = Event()
        
        # Expected configuration
        self.expected_clients = self.config.get("num_clients", 4)
        self.round_duration = self.config.get("round_duration", 300)  # 5 minutes
        
    def on_connect(self, client, userdata, flags, rc):
        logger.info(f"Session manager connected with result code {rc}")
        
        # Subscribe to session management topics
        topics = [
            "hdc/session_status",
            "hdc/client_ready",
            "hdc/training_complete",
            "hdc/aggregation_complete",
            "hdc/inference_ready"
        ]
        
        for topic in topics:
            client.subscribe(topic)
    
    def on_message(self, client, userdata, msg):
        topic = msg.topic
        
        try:
            data = json.loads(msg.payload.decode())
        except json.JSONDecodeError:
            logger.warning(f"Non-JSON message on topic {topic}: {msg.payload.decode()}")
            return
            
        if topic == "hdc/client_ready":
            self.handle_client_ready(data)
            
        elif topic == "hdc/training_complete":
            self.handle_training_complete(data)
            
        elif topic == "hdc/aggregation_complete":
            self.handle_aggregation_complete(data)
            
        elif topic == "hdc/inference_ready":
            self.handle_inference_ready(data)
    
    def handle_client_ready(self, data):
        """Handle client ready notifications"""
        client_id = data.get("client_id")
        self.clients_ready.add(client_id)
        
        logger.info(f"Client {client_id} is ready ({len(self.clients_ready)}/{self.expected_clients})")
        
        if len(self.clients_ready) >= self.expected_clients:
            logger.info("All clients ready. Starting federated learning session...")
            self.start_session()
    
    def handle_training_complete(self, data):
        """Handle training completion"""
        logger.info("Training phase completed")
        self.training_complete.set()
    
    def handle_aggregation_complete(self, data):
        """Handle aggregation completion"""
        logger.info("Aggregation phase completed")
        self.aggregation_complete.set()
    
    def handle_inference_ready(self, data):
        """Handle inference readiness"""
        logger.info("Inference system ready")
        self.complete_session()
    
    def start_session(self):
        """Start the federated learning session"""
        if self.session_active:
            logger.warning("Session already active")
            return
            
        self.session_active = True
        logger.info("=== STARTING HDC FEDERATED LEARNING SESSION ===")
        
        # Session configuration
        session_config = {
            "session_id": f"hdc_session_{int(time.time())}",
            "algorithm": "HDC",
            "num_clients": self.expected_clients,
            "classes": ["bottle", "mouse", "mobile", "sharpner"],
            "hd_dimension": 4096,
            "feature_dimension": 128,
            "start_time": time.time()
        }
        
        # Broadcast session start
        self.mqtt_client.publish("hdc/session_start", json.dumps(session_config))
        
        # Monitor session progress
        self.monitor_session()
    
    def monitor_session(self):
        """Monitor session progress"""
        logger.info("Monitoring session progress...")
        
        # Wait for training completion
        if self.training_complete.wait(timeout=self.round_duration):
            logger.info("✓ Training phase completed successfully")
        else:
            logger.warning("⚠ Training phase timed out")
        
        # Wait for aggregation completion
        if self.aggregation_complete.wait(timeout=60):
            logger.info("✓ Aggregation phase completed successfully")
        else:
            logger.warning("⚠ Aggregation phase timed out")
    
    def complete_session(self):
        """Complete the federated learning session"""
        logger.info("=== HDC FEDERATED LEARNING SESSION COMPLETED ===")
        
        # Session summary
        summary = {
            "session_completed": True,
            "completion_time": time.time(),
            "clients_participated": len(self.clients_ready),
            "training_completed": self.training_complete.is_set(),
            "aggregation_completed": self.aggregation_complete.is_set()
        }
        
        # Broadcast session completion
        self.mqtt_client.publish("hdc/session_complete", json.dumps(summary))
        
        logger.info("Session summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        self.session_active = False
    
    def run(self):
        """Main session loop"""
        logger.info(f"Starting Flotilla HDC Session with config: {self.config_file}")
        logger.info(f"Server endpoint: {self.server_endpoint}")
        
        try:
            # Connect to MQTT
            self.mqtt_client.connect(self.mqtt_host, self.mqtt_port, 60)
            logger.info(f"Connected to MQTT broker at {self.mqtt_host}:{self.mqtt_port}")
            
            # Start MQTT loop in background
            self.mqtt_client.loop_start()
            
            # Wait for clients to be ready
            logger.info(f"Waiting for {self.expected_clients} clients to be ready...")
            
            # Keep session alive
            while self.session_active or len(self.clients_ready) < self.expected_clients:
                time.sleep(1)
            
            # Final wait for completion
            time.sleep(10)
            
        except KeyboardInterrupt:
            logger.info("Session interrupted by user")
        except Exception as e:
            logger.error(f"Session error: {e}")
        finally:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flotilla HDC Session Manager")
    parser.add_argument("config", help="Configuration file path")
    parser.add_argument("--federated_server_endpoint", default="localhost:12345",
                       help="Federated server endpoint")
    
    args = parser.parse_args()
    
    session = FlotillaHDCSession(args.config, args.federated_server_endpoint)
    session.run()