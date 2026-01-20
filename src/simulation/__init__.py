"""
Simulation framework for quantum-safe IoT
"""

from .device import Device, Class0Device, Class1Device, Class2Device
from .gateway import Gateway
from .key_manager import KeyManager

# Optional imports (require ML/FL dependencies)
try:
    from .cloud import Cloud
    from .simulator import Simulator
    __all__ = ['Device', 'Class0Device', 'Class1Device', 'Class2Device', 
               'Gateway', 'Cloud', 'Simulator', 'KeyManager']
except ImportError:
    # ML/FL not available - only basic device-to-gateway functionality
    __all__ = ['Device', 'Class0Device', 'Class1Device', 'Class2Device', 
               'Gateway', 'KeyManager']

