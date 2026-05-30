<img width="3188" height="1202" alt="frame (3)" src="https://github.com/user-attachments/assets/517ad8e9-ad22-457d-9538-a9e62d137cd7" />


# Bald-Rubi_Chan 🎯


## Find Baldness of Head using opencv
### Team Name: Rubi-Chan


### Team Members
- Team Lead: Abdulla Shadin A - Cochin University College of Engineering, Kuttanad
- Member 2: Sivasakthi K - Cochin University College of Engineering, Kuttanad


### Project Description
Baldness Detector is a playful computer vision project that uses deep learning and facial segmentation to estimate scalp coverage in real time. Whether you're flaunting a full mane or embracing the bald aesthetic, this detector has got you covered (or not!).

### The Problem (that doesn't exist)
In a world overflowing with existential crises, climate change, and AI ethics, one issue has somehow slipped through the cracks: spontaneous baldness awareness. Millions walk the streets unaware of their follicle density!

No one asked, but we saw the void. And we filled it with scalp segmentation.

### The Solution (that nobody asked for)
Introducing Baldness Detector — the unsolicited hero of hair diagnostics. Using cutting-edge deep learning and facial geometry, it delivers instant follicle feedback whether you want it or not. Because sometimes, the best innovations are the ones nobody needed, but everyone ends up using at 2 AM with their friends.

## Technical Details
### Technologies/Components Used
For Software:
- Python
- PyTorch, MediaPipe
- OpenCV, NumPy, Torchvision, Git LFS
- BiSeNet (ResNet-18 backbone), Facial Mesh (MediaPipe), Morphological Filters, Custom Threshold Logic
- **NEW: Multi-GPU Acceleration Support** (NVIDIA CUDA, Intel Arc, Intel iGPU)

### Implementation
For Software: VS Code

---

# 🚀 How to Run

## Quick Start (Easiest Way)

### 1. Clone the Repository
```bash
git clone https://github.com/Yoshadin/Bald-Rubi_Chan.git
cd Bald-Rubi_Chan
```

### 2. Choose Your Setup Method

#### **Option A: Automatic GPU Setup (Recommended)**

**Linux/macOS:**
```bash
chmod +x setup_gpu.sh
./setup_gpu.sh
```

**Windows:**
```bash
setup_gpu.bat
```

The setup script will:
- ✅ Detect your GPU automatically (NVIDIA/Intel Arc/Intel iGPU)
- ✅ Install the correct PyTorch version
- ✅ Install all dependencies
- ✅ Verify everything works

#### **Option B: Manual Setup**

**Install dependencies:**
```bash
pip install -r requirements_gpu.txt
```

**Choose GPU type and install PyTorch:**

**For NVIDIA GPU:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For Intel Arc GPU:**
```bash
# First install Intel oneAPI Base Toolkit from:
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html

# Then:
pip install intel-extension-for-pytorch
pip install torch torchvision torchaudio
```

**For Intel Integrated Graphics:**
```bash
# Linux only - Install OpenCL drivers
sudo apt-get install intel-opencl-icd ocl-icd-libopencl1

# Then install PyTorch
pip install torch torchvision torchaudio
```

**For CPU Only:**
```bash
pip install torch torchvision torchaudio
```

### 3. Run the Application

#### **GPU-Optimized Version (Recommended):**
```bash
python main_gpu_optimized.py
```

This version:
- 🔄 Auto-detects your GPU
- ⚡ Uses FP16 acceleration for 2x speedup
- 📊 Shows real-time FPS and performance
- 💾 Displays GPU memory usage

#### **Original CPU Version:**
```bash
python main.py
```

---

## 📊 Performance by GPU Type

| GPU Type | Command | Expected FPS | Setup Difficulty |
|---|---|---|---|
| NVIDIA (RTX 3060+) | `python main_gpu_optimized.py` | 60-100+ | Easy |
| Intel Arc | `python main_gpu_optimized.py` | 40-80 | Medium |
| Intel iGPU | `python main_gpu_optimized.py` | 15-30 | Easy |
| CPU | `python main.py` | 5-10 | Very Easy |

---

## 🎮 Using the Application

### Controls
- **ESC** - Exit the application
- **Camera** - Real-time baldness detection

### Output
The application will display:
- **Face bounding box** - Green rectangle around detected face
- **Baldness percentage** - e.g., "Face 1: 45.2% - Mild thinning"
- **Hair mask overlay** - Color-coded visualization of detected hair

### Baldness Categories
- **< 35%** - No baldness
- **35-45%** - Mild thinning
- **45-60%** - Moderate baldness
- **> 60%** - Bald Boss 👑

---

## 📁 Project Structure

```
Bald-Rubi_Chan/
├── main.py                          # Original CPU version
├── main_gpu_optimized.py            # GPU-accelerated version (RECOMMENDED)
├── setup_gpu.sh                     # Linux/macOS setup script
├── setup_gpu.bat                    # Windows setup script
├── requirements_gpu.txt             # Python dependencies
├── GPU_README.md                    # GPU setup guide
��── GPU_ACCELERATION.md              # Detailed GPU documentation
├── models/
│   └── bisenet.pth                  # Pre-trained BiSeNet model
├── README.md                        # This file
└── .gitattributes                   # Git LFS configuration
```

---

## 🐛 Troubleshooting

### Camera Not Found
```bash
# Check if camera is available
ls /dev/video*  # Linux
# Windows: Check Device Manager → Cameras
# macOS: Check System Preferences → Security & Privacy → Camera
```

### No Face Detected
- Ensure good lighting
- Face should be clearly visible to camera
- Try moving closer to camera

### Low FPS
- Close other applications
- Check GPU usage: `nvidia-smi` (NVIDIA) or `clinfo` (Intel)
- Try reducing input resolution in the code
- Use GPU version instead of CPU

### CUDA/GPU Not Available
```bash
# Verify GPU installation
python -c "import torch; print(torch.cuda.is_available())"

# For NVIDIA:
nvidia-smi

# For Intel Arc:
python -c "import intel_extension_for_pytorch as ipex; print(torch.xpu.is_available())"

# For Intel iGPU:
python -c "import cv2; print(cv2.ocl.haveOpenCL())"
```

See **`GPU_README.md`** for more detailed troubleshooting.

---

## 📊 Monitoring Performance

### NVIDIA GPU
```bash
# Real-time GPU stats
nvidia-smi -l 1
```

### Intel Arc GPU
```bash
# After sourcing Intel oneAPI environment
clinfo
```

### Intel iGPU
```bash
clinfo
```

### All Systems
```bash
# Monitor FPS in the application window
# Window title shows current accelerator: "Baldness Detector - {GPU Type}"
```

---

## 🔄 Performance Optimization

### For Best Performance:
1. **Use GPU version** - `python main_gpu_optimized.py`
2. **Enable FP16** - Automatic for NVIDIA/Intel Arc
3. **Keep drivers updated** - Latest GPU drivers = better performance
4. **Close background apps** - Frees GPU memory
5. **Monitor temperature** - Throttling reduces FPS

### To Adjust Resolution:
Edit `main_gpu_optimized.py` or `main.py`:
```python
# Change from 512 to lower value for faster processing
img_resized = cv2.resize(roi, (384, 384))  # Lower = Faster, Less Accurate
img_resized = cv2.resize(roi, (512, 512))  # Default = Balanced
img_resized = cv2.resize(roi, (640, 640))  # Higher = Slower, More Accurate
```

---

## 📚 Documentation

- **`GPU_README.md`** - Quick reference for GPU setup
- **`GPU_ACCELERATION.md`** - Comprehensive GPU documentation
- **`requirements_gpu.txt`** - Dependencies with GPU options

---

## 🎥 Video Demo

Check out the project demo: [Video Link](https://github.com/user-attachments/assets/4f53ceb2-c19b-4515-b2fb-de921c6fc086)

---

## 📸 Screenshots

<img width="1153" height="784" alt="Screenshot 2025-08-09 175632" src="https://github.com/user-attachments/assets/429a1694-856d-4506-9dfa-64bc1274cefd" />
<img width="697" height="755" alt="Screenshot 2025-08-09 174905" src="https://github.com/user-attachments/assets/18ae0a11-0623-45d6-9a2d-7c794709fd29" />

---

## 🤝 Team Contributions
- Abdulla Shadin A: Coding & GPU Acceleration
- Sivasakthi K: Ideas, Innovations and Library Integration

---

## 📝 License

Open source project - Feel free to use and modify!

---

## 🎉 What's New in v2.0

✨ **Multi-GPU Support Added!**
- ⚡ NVIDIA CUDA acceleration (2x speedup with FP16)
- 🔷 Intel Arc GPU support (oneAPI)
- 💻 Intel Integrated Graphics support (OpenCL)
- 🔄 Automatic GPU detection
- 📊 Real-time performance monitoring
- 🛠️ One-click setup scripts (Linux/macOS/Windows)

---

Made with ❤️ at TinkerHub Useless Projects

![Static Badge](https://img.shields.io/badge/TinkerHub-24?color=%23000000&link=https%3A%2F%2Fwww.tinkerhub.org%2F)
![Static Badge](https://img.shields.io/badge/UselessProjects--25-25?link=https%3A%2F%2Fwww.tinkerhub.org%2Fevents%2FQ2Q1TQKX6Q%2FUseless%2520Projects)

---

## ✨ Quick Reference Card

```
🚀 QUICKEST START:
   1. git clone https://github.com/Yoshadin/Bald-Rubi_Chan.git
   2. cd Bald-Rubi_Chan
   3. ./setup_gpu.sh          (Linux/macOS)
      OR setup_gpu.bat        (Windows)
   4. python main_gpu_optimized.py

🎮 CONTROLS:
   ESC - Exit
   
📊 EXPECTED PERFORMANCE:
   NVIDIA GPU:     60-100+ FPS
   Intel Arc:      40-80 FPS
   Intel iGPU:     15-30 FPS
   CPU:            5-10 FPS

🆘 HELP:
   GPU issues?     → See GPU_README.md
   Setup help?     → Run setup script again
   Performance?    → Check GPU_ACCELERATION.md
```
