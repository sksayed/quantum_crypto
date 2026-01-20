"""
Example: Device-to-Gateway Communication using MQTT with PQC Encryption

This example demonstrates:
1. Setting up key management (fixed at init)
2. Creating devices (Class 0 and Class 1)
3. Creating gateway with MQTT listener
4. Devices publishing encrypted data to gateway
5. Gateway receiving and decrypting data
6. Analysis of encryption metrics
"""

import time
import json
import sys
import os
# Add project root to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation.device import Class0Device, Class1Device
from src.simulation.gateway import Gateway
from src.simulation.key_manager import KeyManager


def main():
    print("=== MQTT Device-to-Gateway Communication Example ===\n")
    
    # Step 1: Initialize Key Manager (fixed at init)
    print("1. Initializing Key Manager...")
    key_manager = KeyManager()
    
    # Step 2: Create Gateway
    print("2. Creating Gateway...")
    gateway = Gateway("GW-1", key_manager=key_manager)
    
    # Register gateway's Kyber keypair
    # Note: Gateway variant should match or be compatible with device variants
    # For Class 1 devices (Kyber-512), gateway can use Kyber-512 or higher
    # Using Kyber-512 to match the Class 1 device for this example
    gateway_kyber = key_manager.register_gateway_kyber("GW-1", variant='kyber512')
    print(f"   Gateway Kyber keypair registered (variant: kyber512)")
    
    # Step 3: Create Devices
    print("3. Creating Devices...")
    
    # Class 0 device (AES-256-GCM)
    device0 = Class0Device("C0-1", battery_percent=80.0)
    aes_key = key_manager.register_device_aes("C0-1")
    print(f"   Class 0 device C0-1 created (AES-256-GCM, key registered)")
    
    # Class 1 device (Kyber-512)
    device1 = Class1Device("C1-1", battery_percent=75.0)
    public_key, private_key = key_manager.register_device_kyber("C1-1", variant='kyber512')
    print(f"   Class 1 device C1-1 created (Kyber-512, keypair registered)")
    
    # Step 4: Start Gateway MQTT Listener
    print("\n4. Starting Gateway MQTT Listener...")
    try:
        gateway.start_mqtt_listener()
        time.sleep(1)  # Wait for connection
        print("   Gateway MQTT listener started successfully")
    except Exception as e:
        print(f"   Error: {e}")
        print("   Make sure MQTT broker (e.g., Mosquitto) is running on localhost:1883")
        return
    
    # Step 5: Devices send data
    print("\n5. Devices sending encrypted data to Gateway...")
    
    # Device 0 sends sensor data (AES-256-GCM)
    sensor_data_0 = {
        "device_id": "C0-1",
        "temperature": 25.5,
        "humidity": 60.2,
        "battery": 80.0,
        "timestamp": time.time()
    }
    
    print(f"\n   Device C0-1 sending data (AES-256-GCM)...")
    result0 = device0.publish_to_gateway(sensor_data_0, key_manager=key_manager)
    print(f"   Result: Success={result0['success']}, "
          f"Encrypt Time={result0['encrypt_time_ms']:.2f}ms, "
          f"Energy={result0['energy_consumed_uj']:.2f}μJ, "
          f"Overhead={result0['overhead_percent']:.1f}%")
    
    time.sleep(0.5)  # Brief delay
    
    # Device 1 sends sensor data (Kyber-512)
    sensor_data_1 = {
        "device_id": "C1-1",
        "temperature": 26.1,
        "humidity": 58.5,
        "battery": 75.0,
        "timestamp": time.time()
    }
    
    print(f"\n   Device C1-1 sending data (Kyber-512)...")
    result1 = device1.publish_to_gateway(sensor_data_1, key_manager=key_manager)
    print(f"   Result: Success={result1['success']}, "
          f"Encrypt Time={result1['encrypt_time_ms']:.2f}ms, "
          f"Energy={result1['energy_consumed_uj']:.2f}μJ, "
          f"Overhead={result1['overhead_percent']:.1f}%")
    
    # Wait for gateway to process messages
    time.sleep(2)
    
    # Step 6: Get Analysis
    print("\n6. Encryption Analysis:")
    print("=" * 60)
    
    # Device metrics
    print("\nDevice Metrics:")
    metrics0 = device0.get_metrics()
    print(f"  C0-1 (AES-256-GCM):")
    print(f"    Messages Sent: {metrics0['messages_sent']}")
    print(f"    Success Rate: {metrics0['success_rate_percent']:.1f}%")
    print(f"    Avg Encrypt Time: {metrics0['avg_encrypt_time_ms']:.2f}ms")
    print(f"    Total Energy: {metrics0['total_energy_uj']:.2f}μJ")
    print(f"    Overhead: {metrics0['overhead_percent']:.1f}%")
    
    metrics1 = device1.get_metrics()
    print(f"\n  C1-1 (Kyber-512):")
    print(f"    Messages Sent: {metrics1['messages_sent']}")
    print(f"    Success Rate: {metrics1['success_rate_percent']:.1f}%")
    print(f"    Avg Encrypt Time: {metrics1['avg_encrypt_time_ms']:.2f}ms")
    print(f"    Total Energy: {metrics1['total_energy_uj']:.2f}μJ")
    print(f"    Overhead: {metrics1['overhead_percent']:.1f}%")
    
    # Gateway analysis
    print("\nGateway Analysis:")
    analysis = gateway.get_encryption_analysis()
    print(f"  Total Communications: {analysis.get('total_communications', 0)}")
    print(f"  Success Rate: {analysis.get('success_rate_percent', 0):.1f}%")
    print(f"  Throughput: {analysis.get('throughput_messages_per_sec', 0):.2f} msg/sec")
    print(f"  Avg Decrypt Time: {analysis.get('avg_decrypt_time_ms', 0):.2f}ms")
    print(f"  Overhead: {analysis.get('overhead_percent', 0):.1f}%")
    
    print("\n  By Crypto Type:")
    for crypto_type, stats in analysis.get('by_crypto_type', {}).items():
        print(f"    {crypto_type}:")
        print(f"      Messages: {stats['count']}")
        print(f"      Avg Decrypt Time: {stats['avg_decrypt_time_ms']:.2f}ms")
    
    print("\n  By Device:")
    for device_id, stats in analysis.get('by_device', {}).items():
        print(f"    {device_id}:")
        print(f"      Messages: {stats['count']}")
        print(f"      Avg Decrypt Time: {stats['avg_decrypt_time_ms']:.2f}ms")
        print(f"      Crypto Types: {list(stats['crypto_types_used'].keys())}")
    
    # Step 7: Cleanup
    print("\n7. Cleaning up...")
    gateway.stop_mqtt_listener()
    print("   Gateway MQTT listener stopped")
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()

