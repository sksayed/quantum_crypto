"""
IoT Device Classes (C0, C1, C2)
"""

import numpy as np
from ..crypto.classical import AES256GCM
from ..crypto.pqc import KyberCrypto


class Device:
    """Base device class"""
    
    def __init__(self, device_id, ram_kb, battery_percent=100.0):
        """
        Initialize device
        
        Args:
            device_id: Unique device identifier
            ram_kb: Available RAM in KB
            battery_percent: Battery level (0-100)
        """
        self.device_id = device_id
        self.ram_kb = ram_kb
        self.battery_percent = battery_percent
        self.crypto_suite = None
        self.energy_consumed = 0.0  # μJ
        self.latency_ms = 0.0
    
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


class Class0Device(Device):
    """Class 0: Ultra-constrained (<10KB RAM) - Uses AES-256-GCM only"""
    
    def __init__(self, device_id, battery_percent=100.0):
        super().__init__(device_id, ram_kb=8.0, battery_percent=battery_percent)
        self.crypto_suite = AES256GCM()
    
    def select_crypto(self, crypto_id):
        """Class 0 always uses AES-256"""
        if crypto_id != 0:
            print(f"Warning: Class 0 device {self.device_id} forced to use AES-256")
        self.crypto_suite = AES256GCM()


class Class1Device(Device):
    """Class 1: Constrained (10-50KB RAM) - Uses Kyber-512 + FL"""
    
    def __init__(self, device_id, battery_percent=100.0):
        super().__init__(device_id, ram_kb=32.0, battery_percent=battery_percent)
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

