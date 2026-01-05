# Quantum-Safe IoT Framework - Simulation Implementation

This is a simulation-based implementation of the quantum-safe IoT framework with Federated Learning and TinyML for HNDL anomaly detection, as described in the paper.

## Architecture Overview

- **Device Layer**: Class 0 (AES-256-GCM), Class 1 (Kyber-512 + FL)
- **Gateway Layer**: TinyML HNDL detector + FL aggregator + Kyber-768
- **Cloud Layer**: FL server + threat intelligence + Kyber-1024

## Windows Setup Instructions

### Prerequisites

1. **Python 3.9+** (recommended: Python 3.10 or 3.11)
   - Download from [python.org](https://www.python.org/downloads/)
   - **Important**: Check "Add Python to PATH" during installation

2. **Git** (optional, for cloning)
   - Download from [git-scm.com](https://git-scm.com/download/win)

### Quick Installation (Automated)

**Option 1: Double-click `setup_windows.bat`**

**Option 2: Run in PowerShell:**
```powershell
.\setup_windows.bat
```

### Manual Installation Steps

1. **Open PowerShell or Command Prompt** in this directory

2. **Create a virtual environment** (recommended):
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   (If you get an execution policy error, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Note on Post-Quantum Cryptography**:
   - The code uses `pyoqs` if available, but includes a fallback simulation mode
   - On Windows, `pyoqs` may not install easily; the fallback will work fine for simulation
   - The fallback simulates Kyber operations with appropriate timing/energy estimates

### Project Structure

```
implementing_quantum_crypto/
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── __init__.py
│   ├── crypto/
│   │   ├── __init__.py
│   │   ├── pqc.py          # Post-Quantum Cryptography (Kyber)
│   │   └── classical.py    # AES-256-GCM
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── hndl_detector.py    # TinyML HNDL anomaly detector
│   │   └── crypto_selector.py  # FL-based crypto selection model
│   ├── fl/
│   │   ├── __init__.py
│   │   ├── client.py       # FL client for devices
│   │   └── server.py        # FL server/aggregator
│   └── simulation/
│       ├── __init__.py
│       ├── device.py        # Device classes (C0, C1, C2)
│       ├── gateway.py       # Gateway with TinyML detector
│       ├── cloud.py         # Cloud FL server
│       └── simulator.py     # Main simulation framework
├── data/
│   └── hndl_dataset.py      # Synthetic HNDL dataset generator
├── models/
│   └── (trained models will be saved here)
└── examples/
    └── run_simulation.py    # Example simulation script
```

## Quick Start

### Activate Virtual Environment First
```powershell
.\venv\Scripts\Activate.ps1
```

### Run Complete Simulation (Recommended)
This single command does everything:
```powershell
python examples\run_simulation.py
```

This will:
1. Generate synthetic HNDL dataset
2. Train the TinyML HNDL detector (~15KB model)
3. Train the Federated Learning crypto-selection model
4. Run the full simulation with 20 C0 devices, 10 C1 devices, 2 gateways

### Individual Steps (Optional)

1. **Generate synthetic HNDL dataset**:
   ```powershell
   python -m data.hndl_dataset
   ```

2. **Train the TinyML HNDL detector** (standalone):
   ```powershell
   python -c "from src.ml.hndl_detector import HNDLDetector; from data.hndl_dataset import generate_hndl_dataset; from sklearn.model_selection import train_test_split; X, y = generate_hndl_dataset(); X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2); d = HNDLDetector(); d.train(X_train, y_train, X_test, y_test); d.save_model('models/hndl_detector.h5')"
   ```

## Usage Examples

See `examples/run_simulation.py` for a complete simulation example.

## Notes

- This is a **simulation** framework. Real hardware deployment would require additional porting.
- Energy measurements are estimated based on cycle counts and power models.
- The HNDL detector uses synthetic data; real deployment requires domain-specific training data.

