"""
Federated Learning Server/Aggregator
Aggregates gradients from clients and updates global model
"""

import numpy as np
from ..ml.crypto_selector import CryptoSelector


class FLServer:
    """
    FL server that aggregates client updates using FedAvg
    """
    
    def __init__(self):
        """Initialize FL server"""
        self.global_model = CryptoSelector()
        self.clients = []
        self.round = 0
        self.convergence_history = []
    
    def add_client(self, client):
        """Add a client to the federation"""
        self.clients.append(client)
    
    def federated_averaging(self, client_weights_list, sample_sizes=None):
        """
        Federated Averaging (FedAvg) aggregation
        
        Args:
            client_weights_list: List of weight arrays from clients
            sample_sizes: Number of samples per client (for weighted averaging)
            
        Returns:
            Aggregated weights
        """
        if sample_sizes is None:
            # Equal weighting
            sample_sizes = [1.0] * len(client_weights_list)
        
        total_samples = sum(sample_sizes)
        
        # Initialize aggregated weights
        aggregated_weights = []
        for i, weights in enumerate(client_weights_list[0]):
            aggregated_weights.append(np.zeros_like(weights))
        
        # Weighted average
        for client_weights, n_samples in zip(client_weights_list, sample_sizes):
            weight = n_samples / total_samples
            for i, w in enumerate(client_weights):
                aggregated_weights[i] += weight * w
        
        return aggregated_weights
    
    def train_round(self, epochs_per_client=5):
        """
        Execute one round of federated learning
        
        Args:
            epochs_per_client: Number of local epochs per client
            
        Returns:
            Average loss across clients
        """
        if len(self.clients) == 0:
            return 0.0
        
        # Get global weights
        global_weights = self.global_model.get_weights()
        
        # Train clients locally
        client_weights_list = []
        sample_sizes = []
        
        for client in self.clients:
            # Get local data size
            if client.local_data is not None:
                n_samples = len(client.local_data[0])
            else:
                n_samples = 0
            
            # Local training
            updated_weights = client.train_local(global_weights, epochs=epochs_per_client)
            client_weights_list.append(updated_weights)
            sample_sizes.append(n_samples)
        
        # Aggregate updates
        aggregated_weights = self.federated_averaging(client_weights_list, sample_sizes)
        
        # Update global model
        self.global_model.set_weights(aggregated_weights)
        
        self.round += 1
        
        # Calculate convergence metric (simplified)
        avg_loss = 0.0  # Would calculate from validation set in practice
        self.convergence_history.append(avg_loss)
        
        return avg_loss
    
    def train(self, num_rounds=50, epochs_per_client=5, convergence_threshold=0.01):
        """
        Train federated model for multiple rounds
        
        Args:
            num_rounds: Maximum number of FL rounds
            epochs_per_client: Local epochs per client per round
            convergence_threshold: Convergence threshold (not used in simplified version)
            
        Returns:
            Training history
        """
        history = []
        
        for round_num in range(num_rounds):
            loss = self.train_round(epochs_per_client)
            history.append(loss)
            
            if round_num % 10 == 0:
                print(f"FL Round {round_num}: Loss = {loss:.4f}")
        
        return history
    
    def get_global_model(self):
        """Get global model"""
        return self.global_model
    
    def save_model(self, model_path):
        """Save global model"""
        self.global_model.save_model(model_path)

