#!/bin/bash
# Multi-Accelerator Setup Script for Baldness Detector
# Supports NVIDIA GPU, Intel Arc GPU, and Intel Integrated Graphics

echo "================================"
echo "Baldness Detector - GPU Setup"
echo "================================"
echo ""

OS_TYPE=$(uname -s)
echo "[INFO] Operating System: $OS_TYPE"
echo ""

# ===== STEP 1: Check for NVIDIA GPU =====
echo "[1/5] Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU found:"
    nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
    HAS_NVIDIA=1
else
    echo "○ No NVIDIA GPU detected"
    HAS_NVIDIA=0
fi
echo ""

# ===== STEP 2: Check for Intel Arc GPU =====
echo "[2/5] Checking for Intel Arc GPU..."
if [ "$OS_TYPE" = "Linux" ] || [ "$OS_TYPE" = "MINGW64_NT" ]; then
    if command -v clinfo &> /dev/null; then
        INTEL_ARC=$(clinfo 2>/dev/null | grep "Intel Arc" || echo "")
        if [ ! -z "$INTEL_ARC" ]; then
            echo "✓ Intel Arc GPU found:"
            clinfo | grep -A 2 "Intel Arc"
            HAS_INTEL_ARC=1
        else
            echo "○ No Intel Arc GPU detected"
            HAS_INTEL_ARC=0
        fi
    else
        echo "○ clinfo not found (Intel Arc detection skipped)"
        HAS_INTEL_ARC=0
    fi
else
    echo "○ Intel Arc GPU detection not available on $OS_TYPE"
    HAS_INTEL_ARC=0
fi
echo ""

# ===== STEP 3: Check for Intel Integrated Graphics =====
echo "[3/5] Checking for Intel Integrated Graphics..."
if [ "$OS_TYPE" = "Linux" ]; then
    if lspci 2>/dev/null | grep -i "vga.*intel" > /dev/null; then
        echo "✓ Intel Integrated Graphics found:"
        lspci | grep -i "vga.*intel"
        HAS_INTEL_IGD=1
    else
        echo "○ No Intel Integrated Graphics detected"
        HAS_INTEL_IGD=0
    fi
elif [ "$OS_TYPE" = "Darwin" ]; then
    echo "✓ Intel/Apple GPU likely available (macOS)"
    HAS_INTEL_IGD=1
else
    echo "○ GPU detection not available on $OS_TYPE"
    HAS_INTEL_IGD=0
fi
echo ""

# ===== STEP 4: User Selection =====
echo "================================"
echo "Select your accelerator:"
echo "================================"
echo ""

if [ $HAS_NVIDIA -eq 1 ]; then
    echo "1) NVIDIA GPU (CUDA)"
fi
if [ $HAS_INTEL_ARC -eq 1 ]; then
    echo "2) Intel Arc GPU (oneAPI)"
fi
if [ $HAS_INTEL_IGD -eq 1 ]; then
    echo "3) Intel Integrated Graphics (OpenCL)"
fi
echo "4) CPU Only"
echo ""

read -p "Enter choice (1-4): " CHOICE
echo ""

# ===== STEP 5: Install based on selection =====
echo "[4/5] Installing PyTorch and dependencies..."

case $CHOICE in
    1)
        if [ $HAS_NVIDIA -eq 0 ]; then
            echo "ERROR: NVIDIA GPU not detected!"
            exit 1
        fi
        echo "Installing for NVIDIA GPU (CUDA 11.8)..."
        pip install --upgrade pip
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ACCELERATOR="NVIDIA GPU"
        ;;
    2)
        if [ $HAS_INTEL_ARC -eq 0 ]; then
            echo "ERROR: Intel Arc GPU not detected!"
            echo "Please install Intel oneAPI Base Toolkit first"
            exit 1
        fi
        echo "Installing for Intel Arc GPU..."
        pip install --upgrade pip
        
        if [ "$OS_TYPE" = "Linux" ]; then
            echo "Initializing Intel oneAPI environment..."
            source /opt/intel/oneapi/setvars.sh
        fi
        
        pip install intel-extension-for-pytorch
        pip install torch torchvision torchaudio
        ACCELERATOR="Intel Arc GPU"
        ;;
    3)
        if [ $HAS_INTEL_IGD -eq 0 ]; then
            echo "ERROR: Intel Integrated Graphics not detected!"
            exit 1
        fi
        echo "Installing for Intel Integrated Graphics..."
        pip install --upgrade pip
        
        if [ "$OS_TYPE" = "Linux" ]; then
            echo "Installing Intel OpenCL drivers..."
            sudo apt-get update
            sudo apt-get install -y intel-opencl-icd ocl-icd-libopencl1
        fi
        
        pip install torch torchvision torchaudio
        ACCELERATOR="Intel Integrated Graphics (iGPU)"
        ;;
    4)
        echo "Installing for CPU only..."
        pip install --upgrade pip
        pip install torch torchvision torchaudio
        ACCELERATOR="CPU"
        ;;
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac

echo ""
echo "[5/5] Installing other dependencies..."
pip install -r requirements_gpu.txt
echo "✓ All dependencies installed"
echo ""

# ===== Verification =====
echo "================================"
echo "Verifying Installation"
echo "================================"
echo ""

python << 'PYTHON_VERIFY'
import torch
import cv2
import sys

print("✓ PyTorch version:", torch.__version__)
print("✓ OpenCV version:", cv2.__version__)

# Check NVIDIA
if torch.cuda.is_available():
    print("✓ NVIDIA CUDA available:", torch.version.cuda)
    print("✓ GPU count:", torch.cuda.device_count())
    print("✓ GPU 0:", torch.cuda.get_device_name(0))
    print("✓ GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

# Check Intel Arc GPU
try:
    import intel_extension_for_pytorch as ipex
    if torch.xpu.is_available():
        print("✓ Intel Arc GPU available via Intel Extension for PyTorch")
        print("✓ XPU device count:", torch.xpu.device_count())
except ImportError:
    pass

# Check Intel Integrated Graphics (OpenCL)
if cv2.ocl.haveOpenCL():
    print("✓ OpenCV OpenCL available (Intel Integrated Graphics)")
    print("✓ OpenCL Device:", cv2.ocl.Device.getDefault().name())
else:
    print("○ OpenCV OpenCL not available (CPU fallback)")

print()
print("✓ Setup complete!")
PYTHON_VERIFY

echo ""
echo "================================"
echo "Next Steps"
echo "================================"
echo ""
echo "Run the optimized baldness detector:"
echo "  python main_gpu_optimized.py"
echo ""
echo "Monitor GPU usage:"
if [ $HAS_NVIDIA -eq 1 ]; then
    echo "  NVIDIA: nvidia-smi -l 1"
fi
if [ $HAS_INTEL_ARC -eq 1 ]; then
    echo "  Intel Arc: source /opt/intel/oneapi/setvars.sh && clinfo"
fi
echo "  Intel iGPU: clinfo"
echo ""
