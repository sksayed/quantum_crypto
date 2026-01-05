"""
Main Simulation Framework
Simulates the quantum-safe IoT architecture with FL and TinyML
"""

import numpy as np
import time
from .device import Class0Device, Class1Device, Class2Device
from .gateway import Gateway
from .cloud import Cloud
from ..fl.client import FLClient
from ..ml.hndl_detector import HNDLDetector


class Simulator:
    """
    Main simulator for quantum-safe IoT framework
    """
    
    def __init__(self, num_c0=20, num_c1=10, num_gateways=2, use_pretrained=True):
        """
        Initialize simulator
        
        Args:
            num_c0: Number of Class 0 devices
            num_c1: Number of Class 1 devices
            num_gateways: Number of gateways
            use_pretrained: Use pretrained models if available
        """
        self.num_c0 = num_c0
        self.num_c1 = num_c1
        self.num_gateways = num_gateways
        
        # Initialize devices
        self.devices_c0 = []
        self.devices_c1 = []
        self.gateways = []
        self.cloud = None
        
        # FL clients for Class 1 devices
        self.fl_clients = []
        
        # Statistics
        self.stats = {
            'energy_consumed': {},
            'latency_overhead': {},
            'hndl_detections': 0,
            'false_positives': 0,
            'crypto_selections': {0: 0, 1: 0, 2: 0, 3: 0}
        }
        
        self._initialize_devices()
        self._initialize_models(use_pretrained)
    
    def _initialize_devices(self):
        """Initialize all devices, gateways, and cloud"""
        # Class 0 devices
        for i in range(self.num_c0):
            device = Class0Device(f"C0-{i}", battery_percent=np.random.uniform(30, 100))
            self.devices_c0.append(device)
        
        # Class 1 devices
        for i in range(self.num_c1):
            device = Class1Device(f"C1-{i}", battery_percent=np.random.uniform(20, 100))
            self.devices_c1.append(device)
        
        # Gateways
        for i in range(self.num_gateways):
            gateway = Gateway(f"GW-{i}")
            self.gateways.append(gateway)
        
        # Cloud
        self.cloud = Cloud("CLOUD-0")
    
    def _initialize_models(self, use_pretrained):
        """Initialize ML models"""
        # HNDL detector (will be trained if no pretrained model)
        hndl_model_path = 'models/hndl_detector.h5'
        if use_pretrained:
            try:
                hndl_detector = HNDLDetector(hndl_model_path)
                print(f"Loaded pretrained HNDL detector: {hndl_detector.get_model_info()}")
            except:
                print("No pretrained HNDL detector found. Will need training.")
                hndl_detector = HNDLDetector()
        else:
            hndl_detector = HNDLDetector()
        
        # Share detector across gateways
        for gateway in self.gateways:
            gateway.hndl_detector = hndl_detector
        
        # FL clients for Class 1 devices
        for device in self.devices_c1:
            # Generate synthetic local data for FL
            X_local = self._generate_fl_training_data(100)
            y_local = self._generate_fl_labels(X_local)
            
            client = FLClient(device.device_id, (X_local, y_local))
            self.fl_clients.append(client)
            device.fl_client = client
            self.cloud.add_fl_client(client)
    
    def _generate_fl_training_data(self, n_samples):
        """Generate synthetic FL training data"""
        # Context: [RAM, Battery%, Latency, Threat]
        X = np.random.rand(n_samples, 4)
        X[:, 0] = X[:, 0] * 50 + 10  # RAM: 10-60 KB
        X[:, 1] = X[:, 1] * 100  # Battery: 0-100%
        X[:, 2] = X[:, 2] * 100 + 5  # Latency: 5-105 ms
        X[:, 3] = X[:, 3]  # Threat: 0-1
        return X
    
    def _generate_fl_labels(self, X):
        """Generate FL labels based on context rules"""
        y = np.zeros((len(X), 4))  # One-hot: [AES, Kyber512, Kyber768, Kyber1024]
        
        for i, context in enumerate(X):
            ram, battery, latency, threat = context
            
            # Rule-based labeling (simplified)
            if battery < 20:
                y[i, 0] = 1  # AES-256
            elif threat > 0.7:
                if ram > 100:
                    y[i, 3] = 1  # Kyber-1024
                else:
                    y[i, 1] = 1  # Kyber-512
            elif ram < 32:
                y[i, 0] = 1  # AES-256
            elif ram < 100:
                y[i, 1] = 1  # Kyber-512
            else:
                y[i, 2] = 1  # Kyber-768
        
        return y
    
    def train_fl(self, num_rounds=50):
        """Train federated learning model"""
        print(f"\n=== Training Federated Learning Model ({num_rounds} rounds) ===")
        history = self.cloud.train_fl(num_rounds, epochs_per_client=5)
        
        # Distribute global model to gateways
        global_model = self.cloud.get_global_model()
        for gateway in self.gateways:
            gateway.crypto_selector = global_model
        
        print(f"FL training completed. Converged in {len(history)} rounds.")
        return history
    
    def simulate_network_flow(self, device, gateway, is_hndl=False):
        """
        Simulate a network flow from device to gateway
        
        Args:
            device: Source device
            gateway: Target gateway
            is_hndl: Whether this is an HNDL attack flow
            
        Returns:
            dict: Flow features
        """
        if is_hndl:
            # HNDL characteristics: large burst, sustained, unusual patterns
            flow_features = np.array([
                np.random.uniform(5000, 10000),  # Large burst length
                np.random.uniform(0.8, 1.0),  # High inter-arrival variance
                np.random.uniform(0.7, 1.0),  # High destination novelty
                np.random.uniform(1000, 5000),  # Long duration
                np.random.uniform(1000, 2000),  # Large payload mean
                np.random.uniform(500, 1000),  # Large payload std
                np.random.uniform(1000, 5000),  # High packet count
                np.random.uniform(10000, 50000)  # High bytes/sec
            ])
        else:
            # Normal flow
            flow_features = np.array([
                np.random.uniform(10, 100),  # Normal burst length
                np.random.uniform(0.1, 0.3),  # Low inter-arrival variance
                np.random.uniform(0.0, 0.2),  # Low destination novelty
                np.random.uniform(10, 100),  # Short duration
                np.random.uniform(50, 200),  # Normal payload mean
                np.random.uniform(10, 50),  # Normal payload std
                np.random.uniform(10, 100),  # Normal packet count
                np.random.uniform(100, 1000)  # Normal bytes/sec
            ])
        
        return flow_features
    
    def run_simulation(self, num_flows=1000, hndl_ratio=0.1):
        """
        Run main simulation
        
        Args:
            num_flows: Number of network flows to simulate
            hndl_ratio: Ratio of HNDL attack flows
        """
        print(f"\n=== Running Simulation ({num_flows} flows, {hndl_ratio*100}% HNDL) ===")
        
        num_hndl = int(num_flows * hndl_ratio)
        num_normal = num_flows - num_hndl
        
        # Generate flows
        flows = []
        for i in range(num_normal):
            device = np.random.choice(self.devices_c0 + self.devices_c1)
            gateway = np.random.choice(self.gateways)
            flows.append((device, gateway, False))
        
        for i in range(num_hndl):
            device = np.random.choice(self.devices_c0 + self.devices_c1)
            gateway = np.random.choice(self.gateways)
            flows.append((device, gateway, True))
        
        np.random.shuffle(flows)
        
        # Process flows
        for flow_idx, (device, gateway, is_hndl) in enumerate(flows):
            # Generate flow features
            flow_features = self.simulate_network_flow(device, gateway, is_hndl)
            
            # Gateway processes flow
            result = gateway.process_flow(device.device_id, flow_features)
            
            # Select crypto based on context
            network_latency = np.random.uniform(5, 50)  # ms
            crypto_id = gateway.select_crypto_for_device(device, network_latency)
            device.select_crypto(crypto_id)
            
            # Update statistics
            self.stats['crypto_selections'][crypto_id] += 1
            
            if result['is_hndl']:
                self.stats['hndl_detections'] += 1
                if not is_hndl:
                    self.stats['false_positives'] += 1
            
            # Simulate energy consumption
            if hasattr(device.crypto_suite, 'get_energy_cost'):
                energy = device.crypto_suite.get_energy_cost()
                if device.device_id not in self.stats['energy_consumed']:
                    self.stats['energy_consumed'][device.device_id] = 0.0
                self.stats['energy_consumed'][device.device_id] += energy
            
            if (flow_idx + 1) % 100 == 0:
                print(f"Processed {flow_idx + 1}/{num_flows} flows...")
        
        print("Simulation completed!")
    
    def print_statistics(self):
        """Print simulation statistics"""
        print("\n=== Simulation Statistics ===")
        print(f"HNDL Detections: {self.stats['hndl_detections']}")
        print(f"False Positives: {self.stats['false_positives']}")
        
        total_energy = sum(self.stats['energy_consumed'].values())
        print(f"Total Energy Consumed: {total_energy:.2f} μJ")
        
        print("\nCrypto Suite Selections:")
        total = sum(self.stats['crypto_selections'].values())
        for crypto_id, count in self.stats['crypto_selections'].items():
            crypto_names = {0: 'AES-256', 1: 'Kyber-512', 2: 'Kyber-768', 3: 'Kyber-1024'}
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {crypto_names[crypto_id]}: {count} ({pct:.1f}%)")
        
        print("\nGateway Statistics:")
        for gateway in self.gateways:
            stats = gateway.get_statistics()
            print(f"  {gateway.gateway_id}: {stats}")

