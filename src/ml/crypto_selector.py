"""
Federated Learning-based Crypto Selection Model
Takes context [RAM, Battery%, Network Latency, Threat Score] and outputs optimal crypto suite
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import pickle


class CryptoSelector:
    """
    ML model for selecting optimal crypto suite based on device context
    Input: [RAM (KB), Battery (%), Network Latency (ms), Threat Score (0-1)]
    Output: Crypto suite selection (0=AES-256, 1=Kyber-512, 2=Kyber-768, 3=Kyber-1024)
    """
    
    def __init__(self, model_path=None):
        """
        Initialize crypto selector
        
        Args:
            model_path: Path to saved model (if None, creates new)
        """
        self.model = None
        self.n_inputs = 4  # RAM, Battery%, Latency, Threat
        self.n_outputs = 4  # 4 crypto options
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._build_model()
    
    def _build_model(self):
        """Build a lightweight model for crypto selection"""
        inputs = keras.Input(shape=(self.n_inputs,), name='context')
        
        # Small network for FL
        x = layers.Dense(16, activation='relu', name='dense1')(inputs)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(8, activation='relu', name='dense2')(x)
        
        # Output: probability distribution over crypto suites
        outputs = layers.Dense(self.n_outputs, activation='softmax', name='crypto_selection')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='crypto_selector')
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """
        Train the crypto selector
        
        Args:
            X_train: Training context features (n_samples, 4)
            y_train: Training labels (one-hot encoded: [AES, Kyber512, Kyber768, Kyber1024])
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history
    
    def select_crypto(self, context):
        """
        Select optimal crypto suite based on context
        
        Args:
            context: Array [RAM (KB), Battery (%), Latency (ms), Threat (0-1)]
            
        Returns:
            tuple: (crypto_id, confidence)
                crypto_id: 0=AES-256, 1=Kyber-512, 2=Kyber-768, 3=Kyber-1024
        """
        if len(context.shape) == 1:
            context = context.reshape(1, -1)
        
        predictions = self.model.predict(context, verbose=0)
        crypto_id = np.argmax(predictions[0])
        confidence = predictions[0][crypto_id]
        
        return int(crypto_id), float(confidence)
    
    def get_weights(self):
        """Get model weights for federated learning"""
        return self.model.get_weights()
    
    def set_weights(self, weights):
        """Set model weights (for FL aggregation)"""
        self.model.set_weights(weights)
    
    def save_model(self, model_path):
        """Save model to file"""
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
        self.model.save(model_path)
    
    def load_model(self, model_path):
        """Load model from file"""
        self.model = keras.models.load_model(model_path)

