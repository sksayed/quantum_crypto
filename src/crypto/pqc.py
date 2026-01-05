"""
Post-Quantum Cryptography: CRYSTALS-Kyber
For Class 1+ devices (constrained to cloud)
"""

import os
import time
import hashlib

try:
    import pyoqs
    PYOQS_AVAILABLE = True
except ImportError:
    PYOQS_AVAILABLE = False
    print("Warning: pyoqs not available. Using fallback implementation.")


class KyberCrypto:
    """
    CRYSTALS-Kyber implementation wrapper
    Supports Kyber-512, Kyber-768, Kyber-1024
    """
    
    # Memory footprints (estimated in KB)
    MEMORY_FOOTPRINT = {
        'kyber512': 8.5,
        'kyber768': 12.0,
        'kyber1024': 18.0
    }
    
    # Energy costs per operation (estimated in μJ)
    ENERGY_COST = {
        'kyber512': 2.5,
        'kyber768': 4.2,
        'kyber1024': 6.8
    }
    
    def __init__(self, variant='kyber512'):
        """
        Initialize Kyber crypto
        
        Args:
            variant: 'kyber512', 'kyber768', or 'kyber1024'
        """
        if variant not in ['kyber512', 'kyber768', 'kyber1024']:
            raise ValueError(f"Invalid variant: {variant}")
        
        self.variant = variant
        self.use_pyoqs = PYOQS_AVAILABLE
        
        if self.use_pyoqs:
            try:
                # Initialize pyoqs
                self.kem = pyoqs.KeyEncapsulation(variant)
                self.public_key = self.kem.generate_keypair()
            except Exception as e:
                print(f"Warning: pyoqs initialization failed: {e}")
                self.use_pyoqs = False
        
        if not self.use_pyoqs:
            # Fallback: simulate Kyber operations
            self.public_key = os.urandom(800 if '512' in variant else (1184 if '768' in variant else 1568))
            self.private_key = os.urandom(1632 if '512' in variant else (2400 if '768' in variant else 3168))
            print(f"Using fallback simulation for {variant}")
    
    def key_exchange_encapsulate(self):
        """
        Generate shared secret and ciphertext (server side)
        
        Returns:
            tuple: (shared_secret, ciphertext, elapsed_time)
        """
        start = time.perf_counter()
        
        if self.use_pyoqs:
            try:
                ciphertext, shared_secret = self.kem.encap_secret(self.public_key)
                elapsed = time.perf_counter() - start
                return shared_secret, ciphertext, elapsed
            except Exception as e:
                print(f"Error in pyoqs encapsulate: {e}")
        
        # Fallback simulation
        shared_secret = os.urandom(32)  # 256-bit shared secret
        ciphertext = os.urandom(768 if '512' in self.variant else (1088 if '768' in self.variant else 1568))
        elapsed = time.perf_counter() - start
        return shared_secret, ciphertext, elapsed
    
    def key_exchange_decapsulate(self, ciphertext):
        """
        Recover shared secret from ciphertext (client side)
        
        Args:
            ciphertext: ciphertext from encapsulate
            
        Returns:
            tuple: (shared_secret, elapsed_time)
        """
        start = time.perf_counter()
        
        if self.use_pyoqs:
            try:
                shared_secret = self.kem.decap_secret(ciphertext)
                elapsed = time.perf_counter() - start
                return shared_secret, elapsed
            except Exception as e:
                print(f"Error in pyoqs decapsulate: {e}")
        
        # Fallback simulation
        shared_secret = os.urandom(32)
        elapsed = time.perf_counter() - start
        return shared_secret, elapsed
    
    def get_memory_footprint(self):
        """Return estimated memory footprint in KB"""
        return self.MEMORY_FOOTPRINT.get(self.variant, 10.0)
    
    def get_energy_cost(self):
        """Return estimated energy cost per key exchange in μJ"""
        return self.ENERGY_COST.get(self.variant, 3.0)
    
    def get_variant(self):
        """Return Kyber variant"""
        return self.variant
    
    def get_security_level(self):
        """Return estimated security level in bits"""
        security = {
            'kyber512': 100,
            'kyber768': 192,
            'kyber1024': 256
        }
        return security.get(self.variant, 100)

