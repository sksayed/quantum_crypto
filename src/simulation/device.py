"""
IoT Device Classes (C0, C1, C2)
"""

import json
import time
import base64
import numpy as np
import paho.mqtt.client as mqtt
from ..crypto.classical import AES256GCM
from ..crypto.pqc import KyberCrypto


class Device:
    """Base device class"""
    
    def __init__(self, device_id, ram_kb, battery_percent=100.0, mqtt_broker_host='localhost', mqtt_broker_port=1883):
        """
        Initialize device
        
        Args:
            device_id: Unique device identifier
            ram_kb: Available RAM in KB
            battery_percent: Battery level (0-100)
            mqtt_broker_host: MQTT broker hostname
            mqtt_broker_port: MQTT broker port
        """
        self.device_id = device_id
        self.ram_kb = ram_kb
        self.battery_percent = battery_percent
        self.crypto_suite = None
        self.energy_consumed = 0.0  # μJ
        self.latency_ms = 0.0
        
        # MQTT setup
        self.mqtt_broker_host = mqtt_broker_host
        self.mqtt_broker_port = mqtt_broker_port
        self.mqtt_client = None
        self.mqtt_connected = False
        
        # Metrics tracking
        self.metrics = {
            'messages_sent': 0,
            'messages_failed': 0,
            'total_encrypt_time_ms': 0.0,
            'total_energy_uj': 0.0,
            'total_ciphertext_bytes': 0,
            'total_plaintext_bytes': 0
        }
    
    def update_battery(self, delta_percent):
        """Update battery level"""
        self.battery_percent = max(0.0, min(100.0, self.battery_percent + delta_percent))
    
    def get_context(self, network_latency_ms=10.0, threat_score=0.0):
        """
        Get device context for crypto selection
        
        Args:
            network_latency_ms: Current network latency
            threat_score: Current threat score (0-1)
            
        Returns:
            Array [RAM, Battery%, Latency, Threat]
        """
        return np.array([
            self.ram_kb,
            self.battery_percent,
            network_latency_ms,
            threat_score
        ])
    
    def select_crypto(self, crypto_id):
        """
        Select crypto suite based on ID
        
        Args:
            crypto_id: 0=AES-256, 1=Kyber-512, 2=Kyber-768, 3=Kyber-1024
        """
        if crypto_id == 0:
            self.crypto_suite = AES256GCM()
        elif crypto_id == 1:
            self.crypto_suite = KyberCrypto('kyber512')
        elif crypto_id == 2:
            self.crypto_suite = KyberCrypto('kyber768')
        elif crypto_id == 3:
            self.crypto_suite = KyberCrypto('kyber1024')
        else:
            raise ValueError(f"Invalid crypto_id: {crypto_id}")
    
    def _init_mqtt_client(self):
        """Initialize MQTT client"""
        if self.mqtt_client is None:
            self.mqtt_client = mqtt.Client(client_id=self.device_id)
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            self.mqtt_connected = True
        else:
            self.mqtt_connected = False
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        self.mqtt_connected = False
    
    def _connect_mqtt(self):
        """Connect to MQTT broker"""
        if not self.mqtt_connected:
            self._init_mqtt_client()
            try:
                self.mqtt_client.connect(self.mqtt_broker_host, self.mqtt_broker_port, 60)
                self.mqtt_client.loop_start()
                time.sleep(0.1)  # Brief wait for connection
            except Exception as e:
                raise ConnectionError(f"Failed to connect to MQTT broker: {e}")
    
    def publish_to_gateway(self, data, key_manager=None, gateway_public_key=None, qos=1):
        """
        Encrypt data and publish to MQTT gateway topic
        
        Args:
            data: Plaintext data (dict, str, or bytes)
            key_manager: KeyManager instance (for AES key or Kyber public key lookup)
            gateway_public_key: Gateway's public key for Kyber (if not using key_manager)
            qos: MQTT QoS level (0, 1, or 2)
        
        Returns:
            dict: Analysis results with encryption metrics
        """
        if self.crypto_suite is None:
            raise ValueError(f"Device {self.device_id} has no crypto suite selected")
        
        # Convert data to bytes
        if isinstance(data, dict):
            data_bytes = json.dumps(data).encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        
        analysis = {
            'device_id': self.device_id,
            'device_class': self.__class__.__name__,
            'crypto_type': None,
            'plaintext_size_bytes': len(data_bytes),
            'encrypt_time_ms': 0.0,
            'energy_consumed_uj': 0.0,
            'ciphertext_size_bytes': 0,
            'overhead_bytes': 0,
            'overhead_percent': 0.0,
            'success': False,
            'timestamp': time.time(),
            'mqtt_published': False
        }
        
        try:
            # Handle AES-256-GCM (symmetric encryption)
            if isinstance(self.crypto_suite, AES256GCM):
                analysis['crypto_type'] = 'AES-256-GCM'
                
                # Encrypt data
                encrypt_start = time.perf_counter()
                nonce, ciphertext, encrypt_time = self.crypto_suite.encrypt(data_bytes)
                encrypt_elapsed = time.perf_counter() - encrypt_start
                
                # Update metrics
                energy = self.crypto_suite.get_energy_cost()
                self.energy_consumed += energy
                
                analysis['encrypt_time_ms'] = encrypt_elapsed * 1000
                analysis['energy_consumed_uj'] = energy
                analysis['ciphertext_size_bytes'] = len(ciphertext) + len(nonce)
                analysis['overhead_bytes'] = analysis['ciphertext_size_bytes'] - len(data_bytes)
                analysis['overhead_percent'] = (analysis['overhead_bytes'] / len(data_bytes)) * 100 if len(data_bytes) > 0 else 0
                
                # Prepare MQTT payload (JSON format)
                mqtt_payload = {
                    'device_id': self.device_id,
                    'crypto_type': 'AES-256-GCM',
                    'nonce': base64.b64encode(nonce).decode('utf-8'),
                    'ciphertext': base64.b64encode(ciphertext).decode('utf-8'),
                    'timestamp': time.time()
                }
                
            # Handle Kyber (key exchange + symmetric encryption)
            elif isinstance(self.crypto_suite, KyberCrypto):
                variant = self.crypto_suite.get_variant()
                analysis['crypto_type'] = f"Kyber-{variant.replace('kyber', '')}"
                
                # Get gateway's public key for encapsulation
                if gateway_public_key is None and key_manager:
                    # For Kyber, we need the gateway's public key to encapsulate
                    # In real implementation, gateway would send its public key during handshake
                    gateway_kyber = key_manager.get_gateway_kyber("GW-1")  # Default gateway ID
                    if gateway_kyber:
                        gateway_public_key = gateway_kyber.public_key
                
                # Step 1: Key exchange (encapsulation) using gateway's public key
                kem_start = time.perf_counter()
                if gateway_public_key:
                    shared_secret, ciphertext_kem, kem_time = self.crypto_suite.key_exchange_encapsulate(gateway_public_key)
                else:
                    # Fallback: use device's own public key (simplified for simulation)
                    shared_secret, ciphertext_kem, kem_time = self.crypto_suite.key_exchange_encapsulate()
                kem_elapsed = time.perf_counter() - kem_start
                
                # Step 2: Derive AES key from shared secret
                aes_key = shared_secret[:32]  # Use first 32 bytes for AES-256
                aes_cipher = AES256GCM(aes_key)
                
                # Step 3: Encrypt data with derived AES key
                encrypt_start = time.perf_counter()
                nonce, ciphertext_data, encrypt_time = aes_cipher.encrypt(data_bytes)
                encrypt_elapsed = time.perf_counter() - encrypt_start
                
                # Update metrics
                kem_energy = self.crypto_suite.get_energy_cost()
                aes_energy = aes_cipher.get_energy_cost()
                total_energy = kem_energy + aes_energy
                self.energy_consumed += total_energy
                
                analysis['encrypt_time_ms'] = (kem_elapsed + encrypt_elapsed) * 1000
                analysis['kem_time_ms'] = kem_elapsed * 1000
                analysis['aes_encrypt_time_ms'] = encrypt_elapsed * 1000
                analysis['energy_consumed_uj'] = total_energy
                analysis['kem_energy_uj'] = kem_energy
                analysis['aes_energy_uj'] = aes_energy
                analysis['ciphertext_size_bytes'] = len(ciphertext_data) + len(nonce) + len(ciphertext_kem)
                analysis['overhead_bytes'] = analysis['ciphertext_size_bytes'] - len(data_bytes)
                analysis['overhead_percent'] = (analysis['overhead_bytes'] / len(data_bytes)) * 100 if len(data_bytes) > 0 else 0
                
                # Prepare MQTT payload (JSON format)
                mqtt_payload = {
                    'device_id': self.device_id,
                    'crypto_type': f'Kyber-{variant.replace("kyber", "")}',
                    'kem_ciphertext': base64.b64encode(ciphertext_kem).decode('utf-8'),
                    'nonce': base64.b64encode(nonce).decode('utf-8'),
                    'ciphertext': base64.b64encode(ciphertext_data).decode('utf-8'),
                    'timestamp': time.time()
                }
            
            else:
                raise ValueError(f"Unsupported crypto suite: {type(self.crypto_suite)}")
            
            # Publish to MQTT
            topic = f"nodes/{self.device_id}/data"
            self._connect_mqtt()
            
            if not self.mqtt_connected:
                raise ConnectionError("MQTT not connected")
            
            mqtt_payload_json = json.dumps(mqtt_payload)
            result = self.mqtt_client.publish(topic, mqtt_payload_json, qos=qos)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                analysis['mqtt_published'] = True
                analysis['success'] = True
                self.metrics['messages_sent'] += 1
                self.metrics['total_encrypt_time_ms'] += analysis['encrypt_time_ms']
                self.metrics['total_energy_uj'] += analysis['energy_consumed_uj']
                self.metrics['total_ciphertext_bytes'] += analysis['ciphertext_size_bytes']
                self.metrics['total_plaintext_bytes'] += len(data_bytes)
            else:
                analysis['mqtt_error'] = f"MQTT publish failed with code {result.rc}"
                self.metrics['messages_failed'] += 1
            
            # Update latency
            self.latency_ms = analysis['encrypt_time_ms']
            
        except Exception as e:
            analysis['error'] = str(e)
            analysis['success'] = False
            self.metrics['messages_failed'] += 1
        
        return analysis
    
    def get_metrics(self):
        """Get device communication metrics"""
        total_messages = self.metrics['messages_sent'] + self.metrics['messages_failed']
        success_rate = (self.metrics['messages_sent'] / total_messages * 100) if total_messages > 0 else 0
        
        avg_encrypt_time = (
            self.metrics['total_encrypt_time_ms'] / self.metrics['messages_sent']
            if self.metrics['messages_sent'] > 0 else 0.0
        )
        
        return {
            'device_id': self.device_id,
            'messages_sent': self.metrics['messages_sent'],
            'messages_failed': self.metrics['messages_failed'],
            'success_rate_percent': success_rate,
            'total_encrypt_time_ms': self.metrics['total_encrypt_time_ms'],
            'avg_encrypt_time_ms': avg_encrypt_time,
            'total_energy_uj': self.metrics['total_energy_uj'],
            'total_ciphertext_bytes': self.metrics['total_ciphertext_bytes'],
            'total_plaintext_bytes': self.metrics['total_plaintext_bytes'],
            'overhead_bytes': self.metrics['total_ciphertext_bytes'] - self.metrics['total_plaintext_bytes'],
            'overhead_percent': (
                (self.metrics['total_ciphertext_bytes'] - self.metrics['total_plaintext_bytes']) /
                self.metrics['total_plaintext_bytes'] * 100
                if self.metrics['total_plaintext_bytes'] > 0 else 0
            ),
            'total_energy_consumed_uj': self.energy_consumed
        }


class Class0Device(Device):
    """Class 0: Ultra-constrained (<10KB RAM) - Uses AES-256-GCM only"""
    
    def __init__(self, device_id, battery_percent=100.0, mqtt_broker_host='localhost', mqtt_broker_port=1883):
        super().__init__(device_id, ram_kb=8.0, battery_percent=battery_percent,
                        mqtt_broker_host=mqtt_broker_host, mqtt_broker_port=mqtt_broker_port)
        self.crypto_suite = AES256GCM()
    
    def select_crypto(self, crypto_id):
        """Class 0 always uses AES-256"""
        if crypto_id != 0:
            print(f"Warning: Class 0 device {self.device_id} forced to use AES-256")
        self.crypto_suite = AES256GCM()


class Class1Device(Device):
    """Class 1: Constrained (10-50KB RAM) - Uses Kyber-512 + FL"""
    
    def __init__(self, device_id, battery_percent=100.0, mqtt_broker_host='localhost', mqtt_broker_port=1883):
        super().__init__(device_id, ram_kb=32.0, battery_percent=battery_percent,
                        mqtt_broker_host=mqtt_broker_host, mqtt_broker_port=mqtt_broker_port)
        self.crypto_suite = KyberCrypto('kyber512')
        self.fl_client = None  # Will be set by simulator
    
    def select_crypto(self, crypto_id):
        """Class 1 can use AES-256 (low battery) or Kyber-512"""
        if crypto_id == 0:
            self.crypto_suite = AES256GCM()
        elif crypto_id == 1:
            self.crypto_suite = KyberCrypto('kyber512')
        else:
            print(f"Warning: Class 1 device {self.device_id} using Kyber-512 (fallback)")
            self.crypto_suite = KyberCrypto('kyber512')


class Class2Device(Device):
    """Class 2: Gateway/Cloud (128MB+ RAM) - Uses Kyber-768/1024"""
    
    def __init__(self, device_id, ram_mb=128, battery_percent=100.0):
        super().__init__(device_id, ram_kb=ram_mb * 1024, battery_percent=battery_percent)
        self.crypto_suite = KyberCrypto('kyber768')
    
    def select_crypto(self, crypto_id):
        """Class 2 can use Kyber-768 or Kyber-1024"""
        if crypto_id == 2:
            self.crypto_suite = KyberCrypto('kyber768')
        elif crypto_id == 3:
            self.crypto_suite = KyberCrypto('kyber1024')
        else:
            print(f"Warning: Class 2 device {self.device_id} using Kyber-768 (fallback)")
            self.crypto_suite = KyberCrypto('kyber768')

