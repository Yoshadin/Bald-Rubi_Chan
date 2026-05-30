@echo off
REM Multi-Accelerator Setup Script for Baldness Detector (Windows)
REM Supports NVIDIA GPU, Intel Arc GPU, and Intel Integrated Graphics

echo ================================
echo Baldness Detector - GPU Setup
echo ================================
echo.

REM ===== STEP 1: Check for NVIDIA GPU =====
echo [1/4] Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ NVIDIA GPU found:
    nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
    set HAS_NVIDIA=1
) else (
    echo o No NVIDIA GPU detected
    set HAS_NVIDIA=0
)
echo.

REM ===== STEP 2: Check for Intel Arc GPU =====
echo [2/4] Checking for Intel Arc GPU...
if exist "C:\Program Files (x86)\Intel\oneAPI" (
    echo ✓ Intel oneAPI detected
    set HAS_INTEL_ARC=1
) else (
    echo o Intel oneAPI not detected
    set HAS_INTEL_ARC=0
)
echo.

REM ===== STEP 3: User Selection =====
echo ================================
echo Select your accelerator:
echo ================================
echo.

if %HAS_NVIDIA% equ 1 (
    echo 1) NVIDIA GPU - CUDA
)
if %HAS_INTEL_ARC% equ 1 (
    echo 2) Intel Arc GPU - oneAPI
)
echo 3) Intel Integrated Graphics / CPU with OpenCL
echo 4) CPU Only
echo.

set /p CHOICE="Enter choice (1-4): "
echo.

REM ===== STEP 4: Install based on selection =====
echo [3/4] Installing PyTorch and dependencies...
echo.

if "%CHOICE%"=="1" (
    if %HAS_NVIDIA% equ 0 (
        echo ERROR: NVIDIA GPU not detected!
        pause
        exit /b 1
    )
    echo Installing for NVIDIA GPU (CUDA 11.8)...
    python -m pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    set ACCELERATOR=NVIDIA GPU
) else if "%CHOICE%"=="2" (
    if %HAS_INTEL_ARC% equ 0 (
        echo ERROR: Intel oneAPI not detected!
        echo Please install Intel oneAPI Base Toolkit first:
        echo https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html
        pause
        exit /b 1
    )
    echo Installing for Intel Arc GPU...
    call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
    python -m pip install --upgrade pip
    pip install intel-extension-for-pytorch
    pip install torch torchvision torchaudio
    set ACCELERATOR=Intel Arc GPU
) else if "%CHOICE%"=="3" (
    echo Installing for Intel Integrated Graphics / CPU with OpenCL...
    python -m pip install --upgrade pip
    pip install torch torchvision torchaudio
    set ACCELERATOR=Intel Integrated Graphics
) else if "%CHOICE%"=="4" (
    echo Installing for CPU only...
    python -m pip install --upgrade pip
    pip install torch torchvision torchaudio
    set ACCELERATOR=CPU
) else (
    echo Invalid choice!
    pause
    exit /b 1
)

echo.
echo [4/4] Installing other dependencies...
pip install -r requirements_gpu.txt
echo ✓ All dependencies installed
echo.

REM ===== Verification =====
echo ================================
echo Verifying Installation
echo ================================
echo.

python -c "import torch; import cv2; print('PyTorch version:', torch.__version__); print('OpenCV version:', cv2.__version__); print('CUDA available:', torch.cuda.is_available()); print('Setup complete!')"

echo.
echo ================================
echo Setup Complete!
echo ================================
echo.
echo Run the optimized baldness detector:
echo   python main_gpu_optimized.py
echo.
echo Monitor GPU usage:
if %HAS_NVIDIA% equ 1 (
    echo   NVIDIA: nvidia-smi -l 1
)
echo.
pause
