"""
Gateway Layer: TinyML HNDL Detector + FL Aggregator + Context-Aware Selection
"""

import json
import time
import base64
import numpy as np
import paho.mqtt.client as mqtt
from ..crypto.pqc import KyberCrypto
from ..crypto.classical import AES256GCM
from .device import Class2Device, Class0Device, Class1Device

# Optional ML imports (for device-to-gateway, these are not required)
try:
    from ..ml.hndl_detector import HNDLDetector
    from ..ml.crypto_selector import CryptoSelector
    ML_AVAILABLE = True
except ImportError:
    HNDLDetector = None
    CryptoSelector = None
    ML_AVAILABLE = False


class Gateway(Class2Device):
    """
    Gateway with AI modules:
    - TinyML HNDL Anomaly Detector
    - FL Aggregator (for crypto selection)
    - Context-aware protocol adaptation
    """
    
    def __init__(self, gateway_id, hndl_detector=None, crypto_selector=None,
                 mqtt_broker_host='localhost', mqtt_broker_port=1883, key_manager=None):
        """
        Initialize gateway
        
        Args:
            gateway_id: Unique gateway identifier
            hndl_detector: Pre-trained HNDL detector (if None, creates new)
            crypto_selector: Pre-trained crypto selector (if None, creates new)
            mqtt_broker_host: MQTT broker hostname
            mqtt_broker_port: MQTT broker port
            key_manager: KeyManager instance for key lookup
        """
        super().__init__(gateway_id, ram_mb=64, battery_percent=100.0)
        self.gateway_id = gateway_id
        
        # AI modules (optional - only if ML is available)
        if ML_AVAILABLE:
            self.hndl_detector = hndl_detector if hndl_detector else HNDLDetector()
            self.crypto_selector = crypto_selector if crypto_selector else CryptoSelector()
        else:
            self.hndl_detector = hndl_detector
            self.crypto_selector = crypto_selector
        
        # Threat tracking
        self.device_threat_scores = {}  # device_id -> threat_score
        self.alert_history = []
        
        # Crypto suite: Kyber-768 for gateway
        self.crypto_suite = KyberCrypto('kyber768')
        
        # Key management
        self.key_manager = key_manager
        
        # MQTT setup
        self.mqtt_broker_host = mqtt_broker_host
        self.mqtt_broker_port = mqtt_broker_port
        self.mqtt_client = None
        self.mqtt_connected = False
        
        # Communication tracking and analysis
        self.communication_log = []
        self.encryption_stats = {
            'total_messages': 0,
            'aes_messages': 0,
            'kyber_messages': 0,
            'total_decrypt_time_ms': 0.0,
            'total_ciphertext_bytes': 0,
            'total_plaintext_bytes': 0,
            'failed_decrypts': 0,
            'by_device': {},
            'by_crypto_type': {}
        }
    
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
        
        if ML_AVAILABLE and self.hndl_detector:
            prob = self.hndl_detector.predict(flow_features)[0]
            is_hndl = prob >= 0.5
        else:
            # Fallback: simple heuristic based on flow features
            prob = 0.0  # No ML, assume no HNDL attack
            is_hndl = False
        
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
        
        # Rule 3: Use ML model for optimal selection (if available)
        if ML_AVAILABLE and self.crypto_selector:
            crypto_id, confidence = self.crypto_selector.select_crypto(context)
        else:
            # Fallback: simple rule-based selection based on device RAM
            if device.ram_kb >= 32:
                crypto_id = 1  # Kyber-512 for Class 1+
            else:
                crypto_id = 0  # AES-256 for Class 0
            confidence = 0.7
        
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
            'hndl_detector_size_kb': self.hndl_detector.get_model_info().get('model_size_kb', 0) if (ML_AVAILABLE and self.hndl_detector) else 0,
            'encryption_stats': self.encryption_stats
        }
    
    def _init_mqtt_client(self):
        """Initialize MQTT client"""
        if self.mqtt_client is None:
            self.mqtt_client = mqtt.Client(client_id=f"{self.gateway_id}_gateway")
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            self.mqtt_client.on_message = self._on_mqtt_message
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            self.mqtt_connected = True
            # Subscribe to all node topics
            client.subscribe("nodes/+/data", qos=1)
            print(f"Gateway {self.gateway_id} connected to MQTT broker and subscribed to nodes/+/data")
        else:
            self.mqtt_connected = False
            print(f"Gateway {self.gateway_id} failed to connect to MQTT broker, return code {rc}")
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        self.mqtt_connected = False
    
    def _on_mqtt_message(self, client, userdata, msg):
        """
        Handle incoming MQTT message from device
        
        Args:
            client: MQTT client
            userdata: User data
            msg: MQTT message object
        """
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            device_id = payload.get('device_id')
            crypto_type = payload.get('crypto_type')
            
            # Process the encrypted message
            result = self._process_encrypted_message(payload)
            
            # Log communication
            log_entry = {
                'device_id': device_id,
                'crypto_type': crypto_type,
                'topic': msg.topic,
                'timestamp': time.time(),
                'result': result
            }
            self.communication_log.append(log_entry)
            
        except Exception as e:
            print(f"Error processing MQTT message: {e}")
    
    def _process_encrypted_message(self, payload):
        """
        Process encrypted message from device
        
        Args:
            payload: Decoded MQTT payload (dict)
        
        Returns:
            dict: Processing result
        """
        device_id = payload.get('device_id')
        crypto_type = payload.get('crypto_type')
        
        result = {
            'device_id': device_id,
            'crypto_type': crypto_type,
            'success': False,
            'decrypt_time_ms': 0.0,
            'hndl_detected': False,
            'threat_score': 0.0,
            'timestamp': time.time()
        }
        
        try:
            decrypt_start = time.perf_counter()
            
            # Handle AES-256-GCM
            if crypto_type == 'AES-256-GCM':
                if not self.key_manager:
                    raise ValueError("KeyManager required for AES-256-GCM decryption")
                
                # Get device's AES key
                aes_key = self.key_manager.get_device_aes_key(device_id)
                if not aes_key:
                    raise ValueError(f"No AES key found for device {device_id}")
                
                # Decode base64
                nonce = base64.b64decode(payload['nonce'])
                ciphertext = base64.b64decode(payload['ciphertext'])
                
                # Decrypt
                aes_cipher = AES256GCM(aes_key)
                plaintext, decrypt_time = aes_cipher.decrypt(nonce, ciphertext)
                
                decrypt_elapsed = time.perf_counter() - decrypt_start
                result['decrypt_time_ms'] = decrypt_elapsed * 1000
                result['plaintext_size_bytes'] = len(plaintext)
                
                # Extract flow features from decrypted data
                flow_features = self._extract_flow_features_from_data(plaintext, crypto_type)
                
            # Handle Kyber
            elif crypto_type and crypto_type.startswith('Kyber-'):
                if not self.key_manager:
                    raise ValueError("KeyManager required for Kyber decryption")
                
                # Get gateway's Kyber instance (with private key)
                gateway_kyber = self.key_manager.get_gateway_kyber(self.gateway_id)
                if not gateway_kyber:
                    raise ValueError(f"No Kyber keypair found for gateway {self.gateway_id}")
                
                # Decode base64
                kem_ciphertext = base64.b64decode(payload['kem_ciphertext'])
                nonce = base64.b64decode(payload['nonce'])
                ciphertext_data = base64.b64decode(payload['ciphertext'])
                
                # Decapsulate to recover shared secret
                shared_secret, decap_time = gateway_kyber.key_exchange_decapsulate(kem_ciphertext)
                
                # Derive AES key
                aes_key = shared_secret[:32]
                aes_cipher = AES256GCM(aes_key)
                
                # Decrypt data
                plaintext, decrypt_time = aes_cipher.decrypt(nonce, ciphertext_data)
                
                decrypt_elapsed = time.perf_counter() - decrypt_start
                result['decrypt_time_ms'] = decrypt_elapsed * 1000
                result['plaintext_size_bytes'] = len(plaintext)
                
                # Extract flow features
                flow_features = self._extract_flow_features_from_data(plaintext, crypto_type)
            
            else:
                raise ValueError(f"Unsupported crypto_type: {crypto_type}")
            
            # Process flow for HNDL detection
            hndl_result = self.process_flow(device_id, flow_features)
            
            result['success'] = True
            result['hndl_detected'] = hndl_result['is_hndl']
            result['threat_score'] = hndl_result['threat_score']
            
            # Update statistics
            self.encryption_stats['total_messages'] += 1
            if crypto_type == 'AES-256-GCM':
                self.encryption_stats['aes_messages'] += 1
            elif crypto_type and crypto_type.startswith('Kyber-'):
                self.encryption_stats['kyber_messages'] += 1
            
            self.encryption_stats['total_decrypt_time_ms'] += result['decrypt_time_ms']
            self.encryption_stats['total_ciphertext_bytes'] += len(payload.get('ciphertext', ''))
            self.encryption_stats['total_plaintext_bytes'] += result.get('plaintext_size_bytes', 0)
            
            # Update per-device stats
            if device_id not in self.encryption_stats['by_device']:
                self.encryption_stats['by_device'][device_id] = {
                    'messages': 0,
                    'total_decrypt_time_ms': 0.0,
                    'crypto_types': {}
                }
            
            device_stats = self.encryption_stats['by_device'][device_id]
            device_stats['messages'] += 1
            device_stats['total_decrypt_time_ms'] += result['decrypt_time_ms']
            
            if crypto_type not in device_stats['crypto_types']:
                device_stats['crypto_types'][crypto_type] = 0
            device_stats['crypto_types'][crypto_type] += 1
            
            # Update per-crypto-type stats
            if crypto_type not in self.encryption_stats['by_crypto_type']:
                self.encryption_stats['by_crypto_type'][crypto_type] = {
                    'messages': 0,
                    'total_decrypt_time_ms': 0.0
                }
            
            crypto_stats = self.encryption_stats['by_crypto_type'][crypto_type]
            crypto_stats['messages'] += 1
            crypto_stats['total_decrypt_time_ms'] += result['decrypt_time_ms']
            
        except Exception as e:
            result['error'] = str(e)
            result['success'] = False
            self.encryption_stats['failed_decrypts'] += 1
        
        return result
    
    def _extract_flow_features_from_data(self, data, crypto_type):
        """
        Extract flow features from decrypted data for HNDL detection
        
        Args:
            data: Decrypted data (bytes)
            crypto_type: Type of encryption used
        
        Returns:
            np.array: Flow features
        """
        # Calculate data size
        data_size = len(data) if isinstance(data, bytes) else len(str(data))
        
        # Generate features based on data characteristics
        # In real implementation, these would come from actual network packet analysis
        return np.array([
            min(data_size, 10000),  # burst_length (capped)
            np.random.uniform(0.1, 0.5),  # inter_arrival_var
            np.random.uniform(0.0, 0.3),  # destination_novelty
            np.random.uniform(10, 100),  # duration
            data_size / 10.0,  # payload_mean (based on size)
            data_size / 50.0,  # payload_std
            max(1, data_size // 100),  # packet_count
            data_size * 10  # bytes_per_sec (estimated)
        ])
    
    def start_mqtt_listener(self):
        """Start MQTT listener (subscribe to topics)"""
        self._init_mqtt_client()
        try:
            self.mqtt_client.connect(self.mqtt_broker_host, self.mqtt_broker_port, 60)
            self.mqtt_client.loop_start()
            print(f"Gateway {self.gateway_id} MQTT listener started")
        except Exception as e:
            raise ConnectionError(f"Failed to start MQTT listener: {e}")
    
    def stop_mqtt_listener(self):
        """Stop MQTT listener"""
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            self.mqtt_connected = False
    
    def get_encryption_analysis(self):
        """
        Get comprehensive analysis of device-to-gateway encryption
        
        Returns:
            dict: Analysis statistics
        """
        if not self.communication_log:
            return {'message': 'No communications logged yet'}
        
        total_messages = self.encryption_stats['total_messages']
        success_rate = (
            (total_messages - self.encryption_stats['failed_decrypts']) / total_messages * 100
            if total_messages > 0 else 0
        )
        
        avg_decrypt_time = (
            self.encryption_stats['total_decrypt_time_ms'] / total_messages
            if total_messages > 0 else 0.0
        )
        
        # Calculate throughput (messages per second)
        if len(self.communication_log) > 1:
            time_span = self.communication_log[-1]['timestamp'] - self.communication_log[0]['timestamp']
            throughput = len(self.communication_log) / time_span if time_span > 0 else 0
        else:
            throughput = 0
        
        return {
            'total_communications': len(self.communication_log),
            'encryption_stats': self.encryption_stats,
            'success_rate_percent': success_rate,
            'avg_decrypt_time_ms': avg_decrypt_time,
            'throughput_messages_per_sec': throughput,
            'total_ciphertext_bytes': self.encryption_stats['total_ciphertext_bytes'],
            'total_plaintext_bytes': self.encryption_stats['total_plaintext_bytes'],
            'overhead_bytes': (
                self.encryption_stats['total_ciphertext_bytes'] - 
                self.encryption_stats['total_plaintext_bytes']
            ),
            'overhead_percent': (
                (self.encryption_stats['total_ciphertext_bytes'] - 
                 self.encryption_stats['total_plaintext_bytes']) /
                self.encryption_stats['total_plaintext_bytes'] * 100
                if self.encryption_stats['total_plaintext_bytes'] > 0 else 0
            ),
            'by_crypto_type': self._analyze_by_crypto_type(),
            'by_device': self._analyze_by_device()
        }
    
    def _analyze_by_crypto_type(self):
        """Analyze communications by encryption type"""
        analysis = {}
        for crypto_type, stats in self.encryption_stats['by_crypto_type'].items():
            count = stats['messages']
            analysis[crypto_type] = {
                'count': count,
                'total_decrypt_time_ms': stats['total_decrypt_time_ms'],
                'avg_decrypt_time_ms': stats['total_decrypt_time_ms'] / count if count > 0 else 0
            }
        return analysis
    
    def _analyze_by_device(self):
        """Analyze communications by device"""
        analysis = {}
        for device_id, stats in self.encryption_stats['by_device'].items():
            count = stats['messages']
            analysis[device_id] = {
                'count': count,
                'total_decrypt_time_ms': stats['total_decrypt_time_ms'],
                'avg_decrypt_time_ms': stats['total_decrypt_time_ms'] / count if count > 0 else 0,
                'crypto_types_used': stats['crypto_types']
            }
        return analysis

