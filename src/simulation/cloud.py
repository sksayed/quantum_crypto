"""
Cloud Layer: FL Server + Threat Intelligence + Kyber-1024
"""

from ..fl.server import FLServer
from ..crypto.pqc import KyberCrypto
from .device import Class2Device


class Cloud(Class2Device):
    """
    Cloud server with:
    - FL training server
    - Threat intelligence database
    - Maximum security (Kyber-1024)
    """
    
    def __init__(self, cloud_id):
        """
        Initialize cloud server
        
        Args:
            cloud_id: Unique cloud identifier
        """
        super().__init__(cloud_id, ram_mb=1024, battery_percent=100.0)
        self.cloud_id = cloud_id
        
        # FL server
        self.fl_server = FLServer()
        
        # Maximum security: Kyber-1024
        self.crypto_suite = KyberCrypto('kyber1024')
        
        # Threat intelligence (simplified)
        self.threat_intelligence = {
            'known_malicious_ips': set(),
            'attack_patterns': [],
            'global_threat_level': 0.0
        }
    
    def add_fl_client(self, client):
        """Add FL client to server"""
        self.fl_server.add_client(client)
    
    def train_fl(self, num_rounds=50, epochs_per_client=5):
        """
        Train federated learning model
        
        Args:
            num_rounds: Number of FL rounds
            epochs_per_client: Local epochs per client
            
        Returns:
            Training history
        """
        return self.fl_server.train(num_rounds, epochs_per_client)
    
    def get_global_model(self):
        """Get global FL model"""
        return self.fl_server.get_global_model()
    
    def update_threat_intelligence(self, threat_data):
        """Update threat intelligence database"""
        # Simplified: just store threat data
        if 'malicious_ips' in threat_data:
            self.threat_intelligence['known_malicious_ips'].update(threat_data['malicious_ips'])
        
        if 'threat_level' in threat_data:
            self.threat_intelligence['global_threat_level'] = threat_data['threat_level']

