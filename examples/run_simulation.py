"""
Example: Run Complete Quantum-Safe IoT Simulation
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.simulation.simulator import Simulator
from src.ml.hndl_detector import HNDLDetector
from data.hndl_dataset import generate_hndl_dataset, save_dataset, load_dataset
from sklearn.model_selection import train_test_split


def train_hndl_detector():
    """Train the TinyML HNDL detector"""
    print("=" * 60)
    print("Step 1: Training TinyML HNDL Detector")
    print("=" * 60)
    
    # Generate or load dataset
    dataset_path = 'data/hndl_dataset.npz'
    if os.path.exists(dataset_path):
        print(f"Loading dataset from {dataset_path}...")
        X, y = load_dataset(dataset_path)
    else:
        print("Generating synthetic HNDL dataset...")
        X, y = generate_hndl_dataset(n_samples=10000, hndl_ratio=0.1)
        save_dataset(X, y, dataset_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Create and train detector
    detector = HNDLDetector()
    
    print("\nTraining HNDL detector...")
    history = detector.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=50,
        batch_size=32
    )
    
    # Evaluate
    predictions = detector.predict(X_test)
    predictions_binary = (predictions >= 0.5).astype(int)
    
    accuracy = np.mean(predictions_binary == y_test)
    precision = np.sum((predictions_binary == 1) & (y_test == 1)) / (np.sum(predictions_binary == 1) + 1e-10)
    recall = np.sum((predictions_binary == 1) & (y_test == 1)) / (np.sum(y_test == 1) + 1e-10)
    fpr = np.sum((predictions_binary == 1) & (y_test == 0)) / (np.sum(y_test == 0) + 1e-10)
    
    print(f"\nHNDL Detector Performance:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall: {recall:.2%}")
    print(f"  False Positive Rate: {fpr:.2%}")
    print(f"  Model Size: {detector.get_model_info()['model_size_kb']:.2f} KB")
    
    # Save model
    model_path = 'models/hndl_detector.h5'
    os.makedirs('models', exist_ok=True)
    detector.save_model(model_path)
    print(f"\nModel saved to {model_path}")
    
    return detector


def run_fl_training():
    """Train federated learning model"""
    print("\n" + "=" * 60)
    print("Step 2: Setting up Federated Learning")
    print("=" * 60)
    
    # Create a minimal simulator to set up FL
    sim = Simulator(num_c0=5, num_c1=10, num_gateways=1, use_pretrained=False)
    
    print("\nTraining FL model...")
    history = sim.train_fl(num_rounds=50)
    
    # Save FL model
    fl_model_path = 'models/fl_crypto_selector.h5'
    sim.cloud.fl_server.save_model(fl_model_path)
    print(f"FL model saved to {fl_model_path}")
    
    return sim


def run_full_simulation():
    """Run complete simulation"""
    print("\n" + "=" * 60)
    print("Step 3: Running Full Simulation")
    print("=" * 60)
    
    # Create simulator with pretrained models
    sim = Simulator(
        num_c0=20,
        num_c1=10,
        num_gateways=2,
        use_pretrained=True
    )
    
    # Run simulation
    sim.run_simulation(num_flows=1000, hndl_ratio=0.1)
    
    # Print statistics
    sim.print_statistics()
    
    return sim


def main():
    """Main execution"""
    print("\n" + "=" * 60)
    print("Quantum-Safe IoT Framework - Simulation")
    print("=" * 60)
    
    # Step 1: Train HNDL detector
    detector = train_hndl_detector()
    
    # Step 2: Train FL model
    sim_fl = run_fl_training()
    
    # Step 3: Run full simulation
    sim = run_full_simulation()
    
    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

