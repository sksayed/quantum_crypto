# MQTT Setup Guide

This guide explains how to set up and use the MQTT-based device-to-gateway communication with PQC encryption.

## Prerequisites

1. **MQTT Broker**: You need an MQTT broker running. We recommend [Mosquitto](https://mosquitto.org/).

### Installing Mosquitto

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install mosquitto mosquitto-clients
sudo systemctl start mosquitto
sudo systemctl enable mosquitto
```

**macOS:**
```bash
brew install mosquitto
brew services start mosquitto
```

**Windows:**
Download from: https://mosquitto.org/download/

**Docker:**
```bash
docker run -it -p 1883:1883 -p 9001:9001 eclipse-mosquitto
```

## Architecture

```
IoT Nodes (Class 0/1/2) 
    ↓ [Encrypt with PQC/AES]
    ↓ [MQTT Publish to nodes/{device_id}/data]
MQTT Broker
    ↓ [MQTT Subscribe to nodes/+/data]
Gateway
    ↓ [Decrypt with PQC/AES]
    ↓ [HNDL Detection & Analysis]
Processed Data
```

## Key Management

Keys are **fixed at initialization**:

- **Class 0 devices**: Pre-shared AES-256-GCM keys (derived from master secret + device_id)
- **Class 1+ devices**: Kyber keypairs (generated at registration)

## Usage Example

See `examples/mqtt_device_gateway_example.py` for a complete example.

### Quick Start

```python
from src.simulation.device import Class0Device, Class1Device
from src.simulation.gateway import Gateway
from src.simulation.key_manager import KeyManager

# 1. Initialize key manager
key_manager = KeyManager()

# 2. Create gateway and register its keypair
gateway = Gateway("GW-1", key_manager=key_manager)
key_manager.register_gateway_kyber("GW-1", variant='kyber768')
gateway.start_mqtt_listener()

# 3. Create devices and register keys
device0 = Class0Device("C0-1")
key_manager.register_device_aes("C0-1")

device1 = Class1Device("C1-1")
key_manager.register_device_kyber("C1-1", variant='kyber512')

# 4. Devices send encrypted data
sensor_data = {"temperature": 25.5, "humidity": 60.2}
result = device0.publish_to_gateway(sensor_data, key_manager=key_manager)

# 5. Get analysis
analysis = gateway.get_encryption_analysis()
print(analysis)
```

## MQTT Topics

- **Device publish**: `nodes/{device_id}/data`
- **Gateway subscribe**: `nodes/+/data` (wildcard for all devices)

## Message Format (JSON)

```json
{
  "device_id": "C0-1",
  "crypto_type": "AES-256-GCM",
  "nonce": "<base64_encoded_nonce>",
  "ciphertext": "<base64_encoded_ciphertext>",
  "timestamp": 1234567890
}
```

For Kyber:
```json
{
  "device_id": "C1-1",
  "crypto_type": "Kyber-512",
  "kem_ciphertext": "<base64_encoded_kem>",
  "nonce": "<base64_encoded_nonce>",
  "ciphertext": "<base64_encoded_data>",
  "timestamp": 1234567890
}
```

## Metrics Collected

### Per Device:
- Encryption time (ms)
- Energy consumption (μJ)
- Ciphertext size vs plaintext size (overhead)
- Success/failure rates
- Throughput (messages/second)

### Per Gateway:
- Total communications
- Decryption time
- HNDL detection results
- Statistics by crypto type
- Statistics by device

## Troubleshooting

1. **MQTT Connection Failed**: Make sure Mosquitto is running
   ```bash
   # Check if running
   sudo systemctl status mosquitto
   
   # Test connection
   mosquitto_pub -h localhost -t test -m "hello"
   mosquitto_sub -h localhost -t test
   ```

2. **Decryption Failed**: Check that keys are properly registered in KeyManager

3. **No Messages Received**: Verify gateway is subscribed and devices are publishing to correct topics

## Security Notes

- This is a **simulation** framework
- In production, use proper key exchange protocols
- Consider using MQTT over TLS for transport security
- Implement proper authentication and authorization

