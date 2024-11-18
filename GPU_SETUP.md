# GPU Acceleration Setup Guide

This guide will help you set up GPU acceleration for the RAG Security Knowledge Base system. Following these steps will significantly improve query performance and embedding generation speed.

## Prerequisites

Before starting, ensure you have:
- NVIDIA GPU (CUDA-compatible)
- Linux operating system (Ubuntu/Pop!_OS recommended)
- Python 3.8 or higher
- pip package manager

## Step 1: Install NVIDIA CUDA Toolkit

First, add NVIDIA's package repository and install the CUDA toolkit:

```bash
# Download NVIDIA repository keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb

# Install the keyring
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Update package list
sudo apt update

# Install CUDA toolkit
sudo apt install cuda-toolkit-12-3
```

## Step 2: Install cuDNN

NVIDIA's CUDA Deep Neural Network library (cuDNN) is required for optimal performance:

```bash
# Install cuDNN packages for CUDA 12.x
sudo apt-get install libcudnn9-cuda-12
sudo apt-get install libcudnn9-dev-cuda-12
sudo apt-get install cudnn9-cuda-12-6
```

## Step 3: Install TensorRT

TensorRT provides additional optimizations for deep learning inference:

```bash
# Install TensorRT
sudo apt-get install tensorrt
```

## Step 4: Update Python Dependencies

Install the GPU-enabled version of onnxruntime:

```bash
# Remove CPU-only version if installed
pip uninstall -y onnxruntime

# Install GPU version
pip install onnxruntime-gpu
```

## Step 5: Verify Installation

Create and run this test script to verify GPU acceleration is working:

```python
import chromadb
import time

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./")
collection = client.get_or_create_collection(name="security_knowledge_base")

# Run a test query
start_time = time.time()
results = collection.query(
    query_texts=["test query"],
    n_results=1
)
query_time = (time.time() - start_time) * 1000

print(f"Query time: {query_time:.2f} ms")
```

## Expected Performance

With GPU acceleration properly configured, you should see:
- Median query times under 2ms
- Ability to handle 500+ queries per second
- Significantly faster embedding generation

## Troubleshooting

### Common Issues

1. **Missing CUDA Libraries**
   ```
   Error: libcudnn.so.9: cannot open shared object file
   ```
   Solution: Ensure cuDNN is properly installed with `sudo apt-get install libcudnn9-cuda-12`

2. **TensorRT Warnings**
   ```
   EP Error: Please install TensorRT libraries
   ```
   Solution: Install TensorRT with `sudo apt-get install tensorrt`

3. **CUDA Version Mismatch**
   Solution: Ensure all installed components (CUDA, cuDNN, TensorRT) are for the same CUDA version

### Verifying CUDA Installation

Check CUDA installation:
```bash
nvcc --version
nvidia-smi
```

### Verifying GPU Usage

Monitor GPU usage during queries:
```bash
nvidia-smi -l 1
```

## Distribution-Specific Instructions

### Ubuntu/Pop!_OS (apt-based)

The instructions in the main guide above are for Ubuntu-based distributions.

### Arch Linux (pacman-based)

First, ensure your system is up to date:
```bash
sudo pacman -Syu
```

#### 1. Install NVIDIA Drivers
```bash
# Install NVIDIA drivers if not already installed
sudo pacman -S nvidia nvidia-utils

# Reboot your system after driver installation
sudo reboot
```

#### 2. Install CUDA Toolkit
```bash
# Install CUDA toolkit and development tools
sudo pacman -S cuda cuda-tools

# Optional: Install additional CUDA samples and documentation
sudo pacman -S cuda-samples cuda-documentation
```

#### 3. Install cuDNN
```bash
# Install cuDNN from the community repository
sudo pacman -S cudnn
```

#### 4. Install TensorRT
```bash
# Install TensorRT from the community repository
sudo pacman -S tensorrt
```

#### 5. Set up Environment Variables
Add these lines to your `~/.bashrc` or `~/.zshrc`:
```bash
export PATH=/opt/cuda/bin:$PATH
export LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH
```

Then source your shell configuration:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

#### 6. Update Python Dependencies
```bash
# Remove CPU-only version if installed
pip uninstall -y onnxruntime

# Install GPU version
pip install onnxruntime-gpu
```

#### Arch-specific Troubleshooting

1. **Missing CUDA Command Line Tools**
   ```
   Command 'nvcc' not found
   ```
   Solution: Install CUDA tools
   ```bash
   sudo pacman -S cuda-tools
   ```

2. **Library Path Issues**
   ```
   Error: Cannot find CUDA libraries
   ```
   Solution: Verify CUDA installation and paths
   ```bash
   # Check if CUDA is in library path
   echo $LD_LIBRARY_PATH
   
   # If needed, add CUDA to library path
   export LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH
   ```

3. **NVIDIA Driver Compatibility**
   ```
   Error: NVIDIA driver version mismatch
   ```
   Solution: Ensure driver and CUDA versions match
   ```bash
   # Check NVIDIA driver version
   nvidia-smi
   
   # Check CUDA version
   nvcc --version
   
   # If needed, install specific driver version
   sudo pacman -S nvidia-470xx-dkms  # for older CUDA versions
   ```

4. **Missing Development Tools**
   Solution: Install base development packages
   ```bash
   sudo pacman -S base-devel
   ```

## Performance Testing

Use the provided `gpu_test.py` script to run performance tests:
```bash
python3 gpu_test.py
```

This will run 100 queries and provide detailed performance metrics.

## Additional Resources

- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

## Support

If you encounter any issues:
1. Check the exact error message
2. Verify all components are installed for the same CUDA version
3. Ensure your GPU drivers are up to date
4. Check system logs for any CUDA-related errors
