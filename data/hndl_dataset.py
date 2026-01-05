"""
Synthetic HNDL Dataset Generator
Generates network flow data with HNDL anomalies
"""

import numpy as np
import pandas as pd


def generate_hndl_dataset(n_samples=10000, hndl_ratio=0.1, random_seed=42):
    """
    Generate synthetic HNDL dataset
    
    Args:
        n_samples: Total number of samples
        hndl_ratio: Ratio of HNDL samples
        random_seed: Random seed for reproducibility
        
    Returns:
        tuple: (X, y) where X is features and y is labels (0=normal, 1=HNDL)
    """
    np.random.seed(random_seed)
    
    n_normal = int(n_samples * (1 - hndl_ratio))
    n_hndl = n_samples - n_normal
    
    features = []
    labels = []
    
    # Generate normal flows
    for _ in range(n_normal):
        flow = [
            np.random.uniform(10, 100),  # burst_length
            np.random.uniform(0.1, 0.3),  # inter_arrival_variance
            np.random.uniform(0.0, 0.2),  # destination_novelty
            np.random.uniform(10, 100),  # flow_duration
            np.random.uniform(50, 200),  # payload_mean
            np.random.uniform(10, 50),  # payload_std
            np.random.uniform(10, 100),  # packet_count
            np.random.uniform(100, 1000)  # bytes_per_second
        ]
        features.append(flow)
        labels.append(0)
    
    # Generate HNDL attack flows
    for _ in range(n_hndl):
        flow = [
            np.random.uniform(5000, 10000),  # Large burst length
            np.random.uniform(0.8, 1.0),  # High inter-arrival variance
            np.random.uniform(0.7, 1.0),  # High destination novelty
            np.random.uniform(1000, 5000),  # Long duration
            np.random.uniform(1000, 2000),  # Large payload mean
            np.random.uniform(500, 1000),  # Large payload std
            np.random.uniform(1000, 5000),  # High packet count
            np.random.uniform(10000, 50000)  # High bytes/sec
        ]
        features.append(flow)
        labels.append(1)
    
    X = np.array(features)
    y = np.array(labels)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


def save_dataset(X, y, filepath='data/hndl_dataset.npz'):
    """Save dataset to file"""
    import os
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    np.savez(filepath, X=X, y=y)
    print(f"Dataset saved to {filepath}")


def load_dataset(filepath='data/hndl_dataset.npz'):
    """Load dataset from file"""
    data = np.load(filepath)
    return data['X'], data['y']


if __name__ == '__main__':
    # Generate and save dataset
    print("Generating HNDL dataset...")
    X, y = generate_hndl_dataset(n_samples=10000, hndl_ratio=0.1)
    save_dataset(X, y)
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Normal samples: {np.sum(y == 0)}, HNDL samples: {np.sum(y == 1)}")

