"""
Gateway Layer: TinyML HNDL Detector + FL Aggregator + Context-Aware Selection
"""

import numpy as np
from ..ml.hndl_detector import HNDLDetector
from ..ml.crypto_selector import CryptoSelector
from ..crypto.pqc import KyberCrypto
from .device import Class2Device


class Gateway(Class2Device):
    """
    Gateway with AI modules:
    - TinyML HNDL Anomaly Detector
    - FL Aggregator (for crypto selection)
    - Context-aware protocol adaptation
    """
    
    def __init__(self, gateway_id, hndl_detector=None, crypto_selector=None):
        """
        Initialize gateway
        
        Args:
            gateway_id: Unique gateway identifier
            hndl_detector: Pre-trained HNDL detector (if None, creates new)
            crypto_selector: Pre-trained crypto selector (if None, creates new)
        """
        super().__init__(gateway_id, ram_mb=64, battery_percent=100.0)
        self.gateway_id = gateway_id
        
        # AI modules
        self.hndl_detector = hndl_detector if hndl_detector else HNDLDetector()
        self.crypto_selector = crypto_selector if crypto_selector else CryptoSelector()
        
        # Threat tracking
        self.device_threat_scores = {}  # device_id -> threat_score
        self.alert_history = []
        
        # Crypto suite: Kyber-768 for gateway
        self.crypto_suite = KyberCrypto('kyber768')
    
    def detect_hndl(self, flow_features):
        """
        Detect HNDL anomaly in network flow
        
        Args:
            flow_features: Array of flow features [burst_length, inter_arrival_var, ...]
            
        Returns:
            tuple: (is_hndl, probability, threat_score)
        """
        if len(flow_features.shape) == 1:
            flow_features = flow_features.reshape(1, -1)
        
        prob = self.hndl_detector.predict(flow_features)[0]
        is_hndl = prob >= 0.5
        
        # Update threat score (exponential moving average)
        threat_score = min(1.0, prob * 1.2)  # Slight amplification
        
        return is_hndl, prob, threat_score
    
    def update_threat_score(self, device_id, threat_score):
        """Update threat score for a device"""
        if device_id in self.device_threat_scores:
            # Exponential moving average
            alpha = 0.3
            self.device_threat_scores[device_id] = (
                alpha * threat_score + (1 - alpha) * self.device_threat_scores[device_id]
            )
        else:
            self.device_threat_scores[device_id] = threat_score
    
    def select_crypto_for_device(self, device, network_latency_ms=10.0):
        """
        Select optimal crypto suite for a device based on context
        
        Args:
            device: Device object
            network_latency_ms: Current network latency
            
        Returns:
            crypto_id: Selected crypto suite ID
        """
        # Get threat score
        threat_score = self.device_threat_scores.get(device.device_id, 0.0)
        
        # Get device context
        context = device.get_context(network_latency_ms, threat_score)
        
        # Context-aware selection rules (from paper)
        # Rule 1: Low battery (<20%) -> use lightweight crypto
        if device.battery_percent < 20.0:
            if isinstance(device, Class2Device):
                return 2  # Kyber-768 (still secure for gateway)
            else:
                return 0  # AES-256 for constrained devices
        
        # Rule 2: High threat (HNDL suspected) -> upgrade to stronger crypto
        if threat_score > 0.7:
            if isinstance(device, Class2Device):
                return 3  # Kyber-1024
            elif device.ram_kb >= 32:
                return 1  # Kyber-512 (best for Class 1)
            else:
                return 0  # AES-256 (fallback)
        
        # Rule 3: Use ML model for optimal selection
        crypto_id, confidence = self.crypto_selector.select_crypto(context)
        
        # Enforce device class constraints
        if isinstance(device, Class0Device):
            crypto_id = 0  # Force AES-256
        elif isinstance(device, Class1Device):
            crypto_id = min(crypto_id, 1)  # Max Kyber-512
        # Class 2 can use any
        
        return crypto_id
    
    def process_flow(self, device_id, flow_features):
        """
        Process network flow: detect HNDL and update threat scores
        
        Args:
            device_id: Source device ID
            flow_features: Flow feature array
            
        Returns:
            dict: Processing results
        """
        is_hndl, prob, threat_score = self.detect_hndl(flow_features)
        
        self.update_threat_score(device_id, threat_score)
        
        if is_hndl:
            self.alert_history.append({
                'device_id': device_id,
                'probability': prob,
                'threat_score': threat_score
            })
        
        return {
            'is_hndl': is_hndl,
            'probability': prob,
            'threat_score': threat_score
        }
    
    def get_statistics(self):
        """Get gateway statistics"""
        return {
            'total_alerts': len(self.alert_history),
            'active_devices': len(self.device_threat_scores),
            'avg_threat_score': np.mean(list(self.device_threat_scores.values())) if self.device_threat_scores else 0.0,
            'hndl_detector_size_kb': self.hndl_detector.get_model_info().get('model_size_kb', 0)
        }

