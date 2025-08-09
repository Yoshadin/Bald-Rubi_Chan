import cv2
import torch
import numpy as np
import mediapipe as mp
import os
from bisenet import BiSeNet
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

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

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True
)

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

    img_resized = cv2.resize(roi, (512, 512), interpolation=cv2.INTER_LINEAR)
    img_t = torch.from_numpy(img_resized.transpose(2, 0, 1)).float().div(255).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    img_t = ((img_t - mean) / std).unsqueeze(0)

    with torch.no_grad():
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

    baldness_history = deque()

    while True:
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
                cv2.imshow("Baldness Detector", overlay)
        else:
            cv2.imshow("Baldness Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("[INFO] ESC pressed. Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Released camera and destroyed all windows")

if __name__ == "__main__":
    main()