# Quick Start Guide - Windows

## Prerequisites Check

1. **Python 3.9+**: Open PowerShell and run:
   ```powershell
   python --version
   ```
   If not installed, download from [python.org](https://www.python.org/downloads/)

2. **pip**: Should come with Python. Verify:
   ```powershell
   pip --version
   ```

## Installation (Choose One Method)

### Method 1: Automated Setup (Recommended)

Double-click `setup_windows.bat` or run in PowerShell:
```powershell
.\setup_windows.bat
```

### Method 2: Manual Setup

1. **Create virtual environment**:
   ```powershell
   python -m venv venv
   ```

2. **Activate virtual environment**:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   (If you get an execution policy error, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

## Running the Simulation

### Step 1: Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### Step 2: Run Full Simulation
```powershell
python examples\run_simulation.py
```

This will:
1. Train the TinyML HNDL detector
2. Train the Federated Learning model
3. Run the complete simulation

### Step 3: View Results

The simulation will output:
- HNDL detection accuracy and false positive rate
- Energy consumption statistics
- Crypto suite selection distribution
- Gateway statistics

## Project Structure

```
implementing_quantum_crypto/
├── src/
│   ├── crypto/          # PQC and classical cryptography
│   ├── ml/              # TinyML models (HNDL detector, crypto selector)
│   ├── fl/              # Federated Learning (client/server)
│   └── simulation/      # Simulation framework
├── data/                # Dataset generation
├── models/              # Saved trained models
└── examples/            # Example scripts
```

## Troubleshooting

### Issue: "pyoqs not available"
- **Solution**: This is expected on Windows. The code will use a fallback simulation mode.

### Issue: TensorFlow installation fails
- **Solution**: Try installing TensorFlow separately:
  ```powershell
  pip install tensorflow
  ```

### Issue: "Execution Policy" error
- **Solution**: Run in PowerShell:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

### Issue: Out of memory during training
- **Solution**: Reduce dataset size in `data/hndl_dataset.py` or reduce batch size in training scripts.

## Next Steps

1. **Modify simulation parameters** in `examples/run_simulation.py`
2. **Experiment with different device configurations** (C0, C1, C2 counts)
3. **Adjust HNDL detection threshold** in `src/simulation/gateway.py`
4. **Customize crypto selection rules** in `src/simulation/gateway.py`

## Expected Output

When running successfully, you should see:
- HNDL detector training progress
- FL training rounds
- Simulation progress (flows processed)
- Final statistics with:
  - Detection accuracy ~90%+
  - False positive rate ~3%
  - Energy consumption metrics
  - Crypto suite distribution

