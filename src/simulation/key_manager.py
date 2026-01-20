"""
Key Management System for Device-Gateway Communication
Fixed key distribution at initialization
"""

import os
import hashlib
from ..crypto.classical import AES256GCM
from ..crypto.pqc import KyberCrypto


class KeyManager:
    """
    Manages cryptographic keys for device-gateway communication.
    Keys are fixed at initialization (pre-shared for AES, keypairs for Kyber).
    """
    
    def __init__(self, master_secret=None):
        """
        Initialize key manager
        
        Args:
            master_secret: Optional master secret for key derivation.
                          If None, generates random master secret.
        """
        if master_secret is None:
            master_secret = os.urandom(32)
        
        self.master_secret = master_secret
        self.device_keys = {}  # device_id -> AES key (for Class 0)
        self.device_keypairs = {}  # device_id -> (public_key, private_key) for Kyber
        self.gateway_kyber_keypairs = {}  # gateway_id -> Kyber keypair for receiving
    
    def register_device_aes(self, device_id, key=None):
        """
        Register a device that uses AES-256-GCM (Class 0)
        
        Args:
            device_id: Device identifier
            key: Pre-shared key (if None, derives from master_secret + device_id)
        
        Returns:
            bytes: The AES key (32 bytes)
        """
        if key is None:
            # Derive key from master secret + device_id
            key_material = self.master_secret + device_id.encode('utf-8')
            key = hashlib.sha256(key_material).digest()
        
        self.device_keys[device_id] = key
        return key
    
    def register_device_kyber(self, device_id, variant='kyber512'):
        """
        Register a device that uses Kyber (Class 1+)
        Generates a keypair for the device
        
        Args:
            device_id: Device identifier
            variant: Kyber variant ('kyber512', 'kyber768', 'kyber1024')
        
        Returns:
            tuple: (public_key, private_key) - public_key is bytes, private_key is KyberCrypto instance
        """
        # Create Kyber instance for device
        device_kyber = KyberCrypto(variant)
        public_key = device_kyber.public_key
        
        # Store keypair
        self.device_keypairs[device_id] = {
            'public_key': public_key,
            'private_key': device_kyber,  # Store the KyberCrypto instance
            'variant': variant
        }
        
        return public_key, device_kyber
    
    def register_gateway_kyber(self, gateway_id, variant='kyber768'):
        """
        Register a gateway with Kyber keypair for receiving messages
        
        Args:
            gateway_id: Gateway identifier
            variant: Kyber variant
        
        Returns:
            KyberCrypto: Gateway's Kyber instance with private key
        """
        gateway_kyber = KyberCrypto(variant)
        self.gateway_kyber_keypairs[gateway_id] = {
            'kyber_instance': gateway_kyber,
            'variant': variant
        }
        return gateway_kyber
    
    def get_device_aes_key(self, device_id):
        """Get AES key for a device"""
        return self.device_keys.get(device_id)
    
    def get_device_kyber_public_key(self, device_id):
        """Get public key for a Kyber device"""
        keypair = self.device_keypairs.get(device_id)
        if keypair:
            return keypair['public_key'], keypair['variant']
        return None, None
    
    def get_gateway_kyber(self, gateway_id):
        """Get gateway's Kyber instance"""
        gateway_data = self.gateway_kyber_keypairs.get(gateway_id)
        if gateway_data:
            return gateway_data['kyber_instance']
        return None

