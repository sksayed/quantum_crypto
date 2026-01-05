"""
TinyML HNDL Anomaly Detector
~15KB model for detecting Harvest-Now-Decrypt-Later anomalies
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import pickle


class HNDLDetector:
    """
    TinyML model for detecting HNDL-like anomalies in network traffic
    Target: ~15KB, ~10K parameters, ~20K MACs
    """
    
    def __init__(self, model_path=None):
        """
        Initialize HNDL detector
        
        Args:
            model_path: Path to saved model (if None, creates new)
        """
        self.model = None
        self.model_size_kb = 0
        self.input_features = [
            'burst_length',
            'inter_arrival_variance',
            'destination_novelty',
            'flow_duration',
            'payload_mean',
            'payload_std',
            'packet_count',
            'bytes_per_second'
        ]
        self.n_features = len(self.input_features)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._build_model()
    
    def _build_model(self):
        """Build a lightweight neural network for HNDL detection"""
        # Input: 8 flow-level features
        inputs = keras.Input(shape=(self.n_features,), name='flow_features')
        
        # Small dense layers (target: ~10K params)
        x = layers.Dense(32, activation='relu', name='dense1')(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(16, activation='relu', name='dense2')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(8, activation='relu', name='dense3')(x)
        
        # Binary classification: HNDL (1) or Normal (0)
        outputs = layers.Dense(1, activation='sigmoid', name='hndl_prob')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='hndl_detector')
        
        # Compile with binary crossentropy
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self._calculate_model_size()
    
    def _calculate_model_size(self):
        """Calculate model size in KB"""
        if self.model is None:
            return
        
        # Save to temporary file to measure size
        temp_path = 'temp_model.h5'
        self.model.save(temp_path)
        size_bytes = os.path.getsize(temp_path)
        self.model_size_kb = size_bytes / 1024.0
        os.remove(temp_path)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """
        Train the HNDL detector
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (0=normal, 1=HNDL)
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
        
        self._calculate_model_size()
        return history
    
    def predict(self, features):
        """
        Predict HNDL probability for flow features
        
        Args:
            features: Array of shape (n_samples, n_features) or (n_features,)
            
        Returns:
            Probability of HNDL anomaly (0-1)
        """
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        predictions = self.model.predict(features, verbose=0)
        return predictions.flatten()
    
    def detect(self, features, threshold=0.5):
        """
        Detect HNDL anomaly (binary classification)
        
        Args:
            features: Flow features
            threshold: Detection threshold (default 0.5)
            
        Returns:
            bool: True if HNDL detected
        """
        prob = self.predict(features)
        return prob[0] >= threshold if isinstance(prob, np.ndarray) else prob >= threshold
    
    def save_model(self, model_path):
        """Save model to file"""
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
        self.model.save(model_path)
        
        # Save metadata
        metadata = {
            'input_features': self.input_features,
            'model_size_kb': self.model_size_kb
        }
        metadata_path = model_path.replace('.h5', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    def load_model(self, model_path):
        """Load model from file"""
        self.model = keras.models.load_model(model_path)
        
        # Load metadata
        metadata_path = model_path.replace('.h5', '_metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.input_features = metadata.get('input_features', self.input_features)
                self.model_size_kb = metadata.get('model_size_kb', 0)
        
        self._calculate_model_size()
    
    def get_model_info(self):
        """Get model information"""
        if self.model is None:
            return {}
        
        total_params = self.model.count_params()
        
        return {
            'model_size_kb': self.model_size_kb,
            'total_params': total_params,
            'input_features': self.n_features,
            'target_size_kb': 15.0
        }
    
    def convert_to_tflite(self, output_path):
        """
        Convert model to TensorFlow Lite for deployment
        
        Args:
            output_path: Path to save .tflite file
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
        
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        tflite_size_kb = len(tflite_model) / 1024.0
        print(f"TFLite model saved: {tflite_size_kb:.2f} KB")
        return tflite_size_kb

