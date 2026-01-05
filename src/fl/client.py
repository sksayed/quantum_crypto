"""
Federated Learning Client
For Class 1 devices participating in FL training
"""

import numpy as np
from ..ml.crypto_selector import CryptoSelector


class FLClient:
    """
    FL client that trains on local data and sends gradients to server
    """
    
    def __init__(self, client_id, local_data=None):
        """
        Initialize FL client
        
        Args:
            client_id: Unique client identifier
            local_data: Tuple (X_local, y_local) for local training
        """
        self.client_id = client_id
        self.model = CryptoSelector()
        self.local_data = local_data
        self.training_rounds = 0
    
    def train_local(self, global_weights, epochs=5, batch_size=32):
        """
        Train model locally on client data
        
        Args:
            global_weights: Global model weights from server
            epochs: Number of local training epochs
            batch_size: Batch size for training
            
        Returns:
            Updated model weights
        """
        # Set global weights
        self.model.set_weights(global_weights)
        
        if self.local_data is None or len(self.local_data[0]) == 0:
            # No local data, return current weights
            return self.model.get_weights()
        
        X_local, y_local = self.local_data
        
        # Local training
        self.model.model.fit(
            X_local, y_local,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        self.training_rounds += 1
        return self.model.get_weights()
    
    def get_model_weights(self):
        """Get current model weights"""
        return self.model.get_weights()
    
    def set_local_data(self, X_local, y_local):
        """Update local training data"""
        self.local_data = (X_local, y_local)

