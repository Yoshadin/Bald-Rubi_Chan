import cv2
import torch
import numpy as np
import mediapipe as mp
import os
from bisenet import BiSeNet
from collections import deque
import time
import platform

# GPU/Accelerator Detection and Setup
device = None
accelerator_type = "CPU"
use_half = False

def detect_accelerator():
    """Detect available GPU/accelerator and set up accordingly"""
    global device, accelerator_type, use_half
    
    print("[INFO] Detecting available accelerators...")
    
    # Check for NVIDIA CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        accelerator_type = "NVIDIA GPU"
        torch.cuda.set_per_process_memory_fraction(0.9)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        try:
            use_half = True  # FP16 for NVIDIA
            print(f"[INFO] Using device: NVIDIA GPU")
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[INFO] CUDA Version: {torch.version.cuda}")
            print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"[INFO] FP16 (half precision) enabled for ~2x speedup")
        except Exception as e:
            print(f"[WARNING] Could not enable FP16: {e}")
            use_half = False
    
    # Check for Intel Arc GPU (oneAPI)
    elif platform.system() in ["Linux", "Windows"]:
        try:
            # Try importing Intel PyTorch Extension (Intel Extension for PyTorch)
            import intel_extension_for_pytorch as ipex
            if torch.xpu.is_available():
                device = torch.device("xpu")
                accelerator_type = "Intel Arc GPU"
                print(f"[INFO] Using device: Intel Arc GPU (XPU)")
                print(f"[INFO] Intel Extension for PyTorch detected")
                use_half = True
            else:
                raise RuntimeError("Intel GPU not available")
        except (ImportError, RuntimeError) as e:
            print(f"[INFO] Intel Arc GPU not available: {e}")
            device = torch.device("cpu")
            accelerator_type = "CPU (Intel Integrated Graphics via OpenCV)"
    else:
        device = torch.device("cpu")
        accelerator_type = "CPU"
    
    if device is None or device.type == "cpu":
        device = torch.device("cpu")
        accelerator_type = "CPU (Intel Integrated Graphics via OpenCV)"
        print(f"[INFO] Using device: CPU with Intel Integrated Graphics acceleration via OpenCV")
    
    print(f"[INFO] Accelerator type: {accelerator_type}\n")

detect_accelerator()

# Load model
model_path = "models/bisenet.pth"
model = BiSeNet(n_classes=19)

try:
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model weights: {e}")
    print("[WARNING] Running with untrained model.")

model.to(device)
model.eval()

# Convert to half precision if available
if use_half and device.type != "cpu":
    try:
        model.half()
        print("[INFO] Model converted to FP16 (half precision)")
    except Exception as e:
        print(f"[WARNING] Could not convert to FP16: {e}")
        use_half = False

# Intel GPU optimization for OpenCV
def setup_opencv_acceleration():
    """Configure OpenCV to use Intel Integrated Graphics"""
    try:
        # Enable OpenCV's GPU acceleration for Intel integrated graphics
        cv2.setUseOptimized(True)
        
        # Try to use OpenCL (works with Intel Integrated Graphics)
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            print("[INFO] OpenCV OpenCL enabled (Intel Integrated Graphics)")
            print(f"[INFO] OpenCL Device: {cv2.ocl.Device.getDefault().name()}")
        else:
            print("[INFO] OpenCL not available, using CPU acceleration")
        
        return True
    except Exception as e:
        print(f"[WARNING] Could not enable OpenCV GPU acceleration: {e}")
        return False

setup_opencv_acceleration()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True
)

# Precompute normalization tensors on device
dtype = torch.float16 if use_half and device.type != "cpu" else torch.float32
mean_tensor = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(3, 1, 1)
std_tensor = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(3, 1, 1)

def expand_face_box(landmarks, orig_h, orig_w):
    xs = np.array([lm.x for lm in landmarks]) * orig_w
    ys = np.array([lm.y for lm in landmarks]) * orig_h
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    w = x_max - x_min
    h = y_max - y_min
    x0 = int(max(0, x_min - 0.25 * w))
    x1 = int(min(orig_w, x_max + 0.25 * w))
    y0 = int(max(0, y_min - 0.85 * h))
    y1 = int(min(orig_h, y_max + 0.15 * h))
    return x0, y0, x1, y1

def get_hair_mask(frame, landmarks, orig_h, orig_w):
    x_min, y_min, x_max, y_max = expand_face_box(landmarks, orig_h, orig_w)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    roi = img[y_min:y_max, x_min:x_max]
    if roi.ndim == 2:
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)

    # Use OpenCV's optimized resize (uses Intel Integrated Graphics if available)
    img_resized = cv2.resize(roi, (512, 512), interpolation=cv2.INTER_LINEAR)
    
    # Convert to tensor and move to device
    img_t = torch.from_numpy(img_resized.transpose(2, 0, 1)).float().to(device)
    
    if use_half and device.type != "cpu":
        img_t = img_t.half()
    
    # Normalize on device
    img_t = img_t.div(255.0)
    img_t = ((img_t - mean_tensor) / std_tensor).unsqueeze(0)

    with torch.no_grad():
        if use_half and device.type != "cpu":
            # Use automatic mixed precision for faster inference
            with torch.cuda.amp.autocast() if device.type == "cuda" else torch.inference_mode():
                raw_out = model(img_t)
        else:
            raw_out = model(img_t)
        
        out = raw_out[0] if isinstance(raw_out, (list, tuple)) else raw_out

    parsing = out.squeeze(0).argmax(0).cpu().numpy()
    unique, counts = np.unique(parsing, return_counts=True)
    print(f"[DEBUG] class histogram: {dict(zip(unique.tolist(), counts.tolist()))}")

    hair_mask_roi = ((parsing == 17).astype(np.uint8) * 255)
    hair_mask_roi = cv2.resize(hair_mask_roi, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_NEAREST)

    full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    full_mask[y_min:y_max, x_min:x_max] = hair_mask_roi

    k3 = np.ones((3, 3), np.uint8)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, k3, iterations=3)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, k3, iterations=1)

    return full_mask, (x_min, y_min, x_max, y_max)

def compute_baldness_from_mask(hair_mask, box, scalp_ratio=0.6):
    x_min, y_min, x_max, y_max = box
    roi = hair_mask[y_min:y_max, x_min:x_max]
    h = roi.shape[0]
    scalp_h = int(h * scalp_ratio)
    scalp_region = roi[:scalp_h, :]

    hair_pixels = np.count_nonzero(scalp_region == 255)
    total_pixels = scalp_region.size

    raw_baldness = (100.0 - (hair_pixels / total_pixels * 100)) if total_pixels > 0 else 0.0

    # Calibration: boost upper end, compress lower
    baldness = min(100.0, (raw_baldness ** 1.12) * 1.25)
    baldness = max(0.0, baldness)

    return baldness

def label_baldness(score):
    if score < 35: return "No baldness"
    elif score < 45: return "Mild thinning"
    elif score < 60: return "Moderate baldness"
    else: return "Bald Boss"

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("[INFO] Camera opened successfully. Starting loop...")
    print(f"[INFO] Running with {accelerator_type}\n")

    baldness_history = deque()
    
    # Performance monitoring
    frame_times = deque(maxlen=30)
    total_frames = 0
    start_time = time.time()

    while True:
        frame_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        orig_h, orig_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for idx, face_landmarks in enumerate(results.multi_face_landmarks):
                landmarks = [lmk for lmk in face_landmarks.landmark]
                hair_mask, box = get_hair_mask(frame, landmarks, orig_h, orig_w)
                baldness = compute_baldness_from_mask(hair_mask, box)

                baldness_history.append(baldness)
                if len(baldness_history) > 5:
                    baldness_history.popleft()
                avg_baldness = sum(baldness_history) / len(baldness_history)
                label = label_baldness(avg_baldness)

                x_min, y_min, x_max, y_max = box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"Face {idx+1}: {avg_baldness:.1f}% - {label}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                colored_mask = cv2.applyColorMap(hair_mask, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(frame, 0.75, colored_mask, 0.25, 0)
                cv2.imshow(f"Baldness Detector - {accelerator_type}", overlay)
        else:
            cv2.imshow(f"Baldness Detector - {accelerator_type}", frame)

        # Performance tracking
        frame_time = time.time() - frame_start
        frame_times.append(frame_time)
        total_frames += 1
        
        # Display performance stats every 30 frames
        if total_frames % 30 == 0:
            avg_time = sum(frame_times) / len(frame_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            elapsed = time.time() - start_time
            print(f"[PERF] Frame {total_frames} | FPS: {fps:.2f} | Avg Time: {avg_time*1000:.2f}ms | Device: {accelerator_type}")
            
            # GPU memory usage for NVIDIA
            if device.type == "cuda":
                gpu_mem = torch.cuda.memory_allocated() / 1e9
                print(f"[GPU] NVIDIA Memory used: {gpu_mem:.2f} GB")
            
            # Intel Arc GPU stats
            if device.type == "xpu":
                try:
                    import intel_extension_for_pytorch as ipex
                    xpu_mem = torch.xpu.memory_allocated() / 1e9
                    print(f"[GPU] Intel Arc Memory used: {xpu_mem:.2f} GB")
                except:
                    pass

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("[INFO] ESC pressed. Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    total_elapsed = time.time() - start_time
    avg_fps = total_frames / total_elapsed if total_elapsed > 0 else 0
    print(f"\n[FINAL STATS]")
    print(f"Accelerator: {accelerator_type}")
    print(f"Total frames: {total_frames}")
    print(f"Total time: {total_elapsed:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    
    print("[INFO] Released camera and destroyed all windows")

if __name__ == "__main__":
    main()
