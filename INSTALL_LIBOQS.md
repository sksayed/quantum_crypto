# Installing liboqs on a New Machine

This guide explains how to set up liboqs (Post-Quantum Cryptography library) when cloning this repository on another PC.

## Why You Need liboqs

The code uses **CRYSTALS-Kyber** for post-quantum cryptography, which requires:
1. **liboqs** - The C library (provides Kyber implementation)
2. **pyoqs** (oqs Python package) - Python bindings for liboqs

**Note:** The `liboqs_install/` directory in this repo contains Linux-compiled libraries that are **platform-specific**. You'll need to build liboqs for your platform.

---

## Installation Methods

### Method 1: Install via Package Manager (Linux - Recommended)

#### Ubuntu/Debian:
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake git libssl-dev

# Clone and build liboqs
git clone https://github.com/open-quantum-safe/liboqs.git
cd liboqs
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
make -j$(nproc)
sudo make install

# Install Python bindings
cd ../..
git clone https://github.com/open-quantum-safe/liboqs-python.git
cd liboqs-python
pip install .
```

#### Fedora/RHEL:
```bash
sudo dnf install gcc cmake git openssl-devel
# Then follow the same build steps as above
```

#### macOS (using Homebrew):
```bash
# Install dependencies
brew install cmake openssl

# Build liboqs
git clone https://github.com/open-quantum-safe/liboqs.git
cd liboqs
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
make -j$(sysctl -n hw.ncpu)
sudo make install

# Install Python bindings
cd ../..
git clone https://github.com/open-quantum-safe/liboqs-python.git
cd liboqs-python
pip install .
```

---

### Method 2: Install to Local Directory (No sudo required)

If you don't have sudo access, install to a local directory:

```bash
# Set installation prefix
export PREFIX=$HOME/local

# Build liboqs
git clone https://github.com/open-quantum-safe/liboqs.git
cd liboqs
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$PREFIX ..
make -j$(nproc)  # or make -j4 on macOS
make install

# Set library path
export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH  # Linux
# OR
export DYLD_LIBRARY_PATH=$PREFIX/lib:$DYLD_LIBRARY_PATH  # macOS

# Install Python bindings
cd ../..
git clone https://github.com/open-quantum-safe/liboqs-python.git
cd liboqs-python
pip install --user .
```

**Important:** Add these to your `~/.bashrc` or `~/.zshrc`:
```bash
export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH  # Linux
export DYLD_LIBRARY_PATH=$HOME/local/lib:$DYLD_LIBRARY_PATH  # macOS
export PKG_CONFIG_PATH=$HOME/local/lib/pkgconfig:$PKG_CONFIG_PATH
```

---

### Method 3: Windows Installation

Windows is more complex. You have two options:

#### Option A: Use WSL (Windows Subsystem for Linux) - Recommended
1. Install WSL2 with Ubuntu
2. Follow the Ubuntu instructions above

#### Option B: Build with Visual Studio (Advanced)
```powershell
# Install Visual Studio 2019+ with C++ tools
# Install CMake: https://cmake.org/download/
# Install Git: https://git-scm.com/download/win

# Open "x64 Native Tools Command Prompt for VS"

git clone https://github.com/open-quantum-safe/liboqs.git
cd liboqs
mkdir build && cd build
cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX=C:\liboqs ..
cmake --build . --config Release
cmake --install . --config Release

# Install Python bindings
cd ..\..
git clone https://github.com/open-quantum-safe/liboqs-python.git
cd liboqs-python
pip install .
```

**Note:** On Windows, the code has a fallback mode if liboqs is not available, but it's simulation-only.

---

## Verify Installation

After installation, verify it works:

```python
python -c "import oqs; kem = oqs.KeyEncapsulation('Kyber512'); print('liboqs installed successfully!')"
```

If you see "liboqs installed successfully!", you're good to go!

---

## Quick Setup Script

Save this as `setup_liboqs.sh` and run it:

```bash
#!/bin/bash
set -e

PREFIX=${1:-$HOME/local}
echo "Installing liboqs to $PREFIX"

# Install dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential cmake git libssl-dev

# Build liboqs
if [ ! -d "liboqs" ]; then
    git clone https://github.com/open-quantum-safe/liboqs.git
fi
cd liboqs
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$PREFIX ..
make -j$(nproc)
make install
cd ../..

# Install Python bindings
if [ ! -d "liboqs-python" ]; then
    git clone https://github.com/open-quantum-safe/liboqs-python.git
fi
cd liboqs-python
pip install --user .

echo "Installation complete!"
echo "Add to ~/.bashrc:"
echo "  export LD_LIBRARY_PATH=$PREFIX/lib:\$LD_LIBRARY_PATH"
echo "  export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig:\$PKG_CONFIG_PATH"
```

Make it executable and run:
```bash
chmod +x setup_liboqs.sh
./setup_liboqs.sh
```

---

## Troubleshooting

### Issue: "liboqs not found" or "Cannot find liboqs"
**Solution:**
```bash
# Linux
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# OR if installed locally
export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH

# macOS
export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH
```

### Issue: "pkg-config cannot find liboqs"
**Solution:**
```bash
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
# OR
export PKG_CONFIG_PATH=$HOME/local/lib/pkgconfig:$PKG_CONFIG_PATH
```

### Issue: Python can't import oqs
**Solution:**
1. Make sure liboqs is installed and library path is set
2. Reinstall pyoqs: `pip uninstall oqs && pip install .` (from liboqs-python directory)
3. Check Python can find the library: `python -c "import ctypes; print(ctypes.util.find_library('oqs'))"`

### Issue: Build fails with "OpenSSL not found"
**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install libssl-dev

# macOS
brew install openssl
export OPENSSL_ROOT_DIR=$(brew --prefix openssl)

# Then rebuild liboqs
```

---

## Alternative: Use Docker (No Installation Needed)

If installation is problematic, use Docker:

```bash
# Create Dockerfile
cat > Dockerfile << EOF
FROM python:3.10

RUN apt-get update && apt-get install -y \\
    build-essential cmake git libssl-dev

RUN git clone https://github.com/open-quantum-safe/liboqs.git && \\
    cd liboqs && mkdir build && cd build && \\
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. && \\
    make -j\$(nproc) && make install

RUN git clone https://github.com/open-quantum-safe/liboqs-python.git && \\
    cd liboqs-python && pip install .

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

CMD ["python", "examples/run_simulation.py"]
EOF

# Build and run
docker build -t quantum-crypto .
docker run -it quantum-crypto
```

---

## Summary

**Minimum steps for a new Linux machine:**
1. Install dependencies: `sudo apt-get install build-essential cmake git libssl-dev`
2. Clone and build liboqs
3. Install Python bindings: `pip install .` (from liboqs-python)
4. Set library path if needed
5. Verify: `python -c "import oqs; print('OK')"`

**For Windows:** Use WSL or expect fallback simulation mode.
