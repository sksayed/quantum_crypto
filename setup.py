"""
Setup script for Quantum-Safe IoT Framework Simulation
"""
from setuptools import setup, find_packages

setup(
    name="quantum-safe-iot",
    version="0.1.0",
    description="Simulation framework for quantum-safe IoT with FL and TinyML",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "tensorflow>=2.14.0",
        "scikit-learn>=1.3.0",
        "cryptography>=41.0.0",
        "flwr>=1.5.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "simpy>=4.0.0",
    ],
)

