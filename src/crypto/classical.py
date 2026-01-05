"""
Classical Cryptography: AES-256-GCM
For Class 0 devices (ultra-constrained, <10KB RAM)
"""

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend
import os
import time


class AES256GCM:
    """AES-256-GCM implementation for Class 0 devices"""
    
    KEY_SIZE = 32  # 256 bits
    NONCE_SIZE = 12  # 96 bits for GCM
    
    def __init__(self, key=None):
        """
        Initialize AES-256-GCM
        
        Args:
            key: 32-byte key (if None, generates random key)
        """
        if key is None:
            key = os.urandom(self.KEY_SIZE)
        elif len(key) != self.KEY_SIZE:
            raise ValueError(f"Key must be {self.KEY_SIZE} bytes")
        
        self.key = key
        self.cipher = AESGCM(key)
        self._memory_footprint = 2.5  # KB (estimated)
        self._energy_per_op = 0.5  # μJ (estimated)
    
    def encrypt(self, plaintext, associated_data=b""):
        """
        Encrypt plaintext using AES-256-GCM
        
        Args:
            plaintext: bytes to encrypt
            associated_data: optional authenticated data
            
        Returns:
            tuple: (nonce, ciphertext, tag)
        """
        nonce = os.urandom(self.NONCE_SIZE)
        start = time.perf_counter()
        ciphertext = self.cipher.encrypt(nonce, plaintext, associated_data)
        elapsed = time.perf_counter() - start
        
        # GCM includes tag in ciphertext
        return nonce, ciphertext, elapsed
    
    def decrypt(self, nonce, ciphertext, associated_data=b""):
        """
        Decrypt ciphertext using AES-256-GCM
        
        Args:
            nonce: nonce used for encryption
            ciphertext: encrypted data (includes tag)
            associated_data: optional authenticated data
            
        Returns:
            tuple: (plaintext, elapsed_time)
        """
        start = time.perf_counter()
        plaintext = self.cipher.decrypt(nonce, ciphertext, associated_data)
        elapsed = time.perf_counter() - start
        return plaintext, elapsed
    
    def get_memory_footprint(self):
        """Return estimated memory footprint in KB"""
        return self._memory_footprint
    
    def get_energy_cost(self, operation="encrypt"):
        """Return estimated energy cost per operation in μJ"""
        return self._energy_per_op
    
    def get_key_size(self):
        """Return key size in bytes"""
        return self.KEY_SIZE

