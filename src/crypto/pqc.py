"""
Post-Quantum Cryptography: CRYSTALS-Kyber
For Class 1+ devices (constrained to cloud)
"""

import os
import time

try:
    import oqs
except ImportError as e:
    # For this project we require a real Kyber implementation rather than a
    # simulated fallback. Make the dependency explicit so errors surface early.
    raise ImportError(
        "The 'oqs' package (liboqs-python) is required for KyberCrypto. "
        "Please install it (and liboqs) before using the PQC module."
    ) from e


class KyberCrypto:
    """
    CRYSTALS-Kyber implementation wrapper
    Supports Kyber-512, Kyber-768, Kyber-1024
    """
    
    # Energy costs per operation (estimated in μJ).
    # These are *model* values for simulation, not hardware measurements.
    # They can be tuned to match a specific device or taken from literature.
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
        # Map variant names to liboqs mechanism names (capitalize first letter)
        mechanism_name = variant[0].upper() + variant[1:]  # 'kyber512' -> 'Kyber512'
        
        # Initialize oqs KEM instance. Any errors here should be explicit,
        # since we no longer provide a simulated fallback.
        self.kem = oqs.KeyEncapsulation(mechanism_name)
        # generate_keypair() returns only the public key (bytes)
        # The private key is stored internally in the KEM object
        self.public_key = self.kem.generate_keypair()
    
    def key_exchange_encapsulate(self, public_key=None):
        """
        Generate shared secret and ciphertext
        
        Args:
            public_key: Public key to encapsulate with (if None, uses own public_key)
        
        Returns:
            tuple: (shared_secret, ciphertext, elapsed_time)
        """
        start = time.perf_counter()
        target_public_key = public_key if public_key is not None else self.public_key
        ciphertext, shared_secret = self.kem.encap_secret(target_public_key)
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
        shared_secret = self.kem.decap_secret(ciphertext)
        elapsed = time.perf_counter() - start
        return shared_secret, elapsed
    
    def get_memory_footprint(self):
        """
        Return an estimated memory footprint in KB.

        This is derived from the public and secret key sizes exposed by oqs
        (algorithm-level footprint), not a full process RSS measurement.
        """
        # Get key sizes from the KEM instance
        # Public key is available directly
        public_key_len = len(self.public_key)
        # For private key size, we need to estimate or use known values
        # Kyber-512: pub=800, priv=1632; Kyber-768: pub=1184, priv=2400; Kyber-1024: pub=1568, priv=3168
        private_key_sizes = {
            'kyber512': 1632,
            'kyber768': 2400,
            'kyber1024': 3168
        }
        private_key_len = private_key_sizes.get(self.variant, 1632)
        bytes_total = public_key_len + private_key_len
        return bytes_total / 1024.0
    
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

#ICML 