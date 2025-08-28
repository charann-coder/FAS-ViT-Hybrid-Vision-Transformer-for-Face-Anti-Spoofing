# Live face anti-spoofing detection with webcam
import cv2
import torch
import argparse
import time
import json
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from model import FASViTClassifier

# ---- Configuration ----
IMG_SIZE = 128
PATCH_SIZE = 4
DEPTH = 1
NUM_SPOOF_TYPES = 3
IS_MULTITASK = False
NORMALIZE = True

# ---- Preprocessing transforms ----
norm = transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)) if NORMALIZE else (lambda x: x)
tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    norm
])

def get_args():
    parser = argparse.ArgumentParser(description="Live Face Anti-Spoofing Detection")
    parser.add_argument("--ckpt", default="checkpoints/epoch_19.pth", help="Path to model checkpoint")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    parser.add_argument("--threshold", type=float, default=0.65, help="Threshold for spoof detection (default 0.65)")
    parser.add_argument("--threshold_json", type=str, default=None, help="JSON with threshold value")
    parser.add_argument("--save_dir", default="captures", help="Directory to save captured frames")
    parser.add_argument("--smooth", type=int, default=5, help="Temporal smoothing window")
    parser.add_argument("--margin", type=float, default=0.25, help="Face margin multiplier")
    parser.add_argument("--min-face", type=int, default=60, help="Min face size (pixels)")
    parser.add_argument("--max-faces", type=int, default=10, help="Max number of faces to track")
    parser.add_argument("--enhance", action="store_true", help="Apply image enhancement")
    parser.add_argument("--adaptive-threshold", action="store_true", help="Adjust threshold based on lighting")
    parser.add_argument("--no-hysteresis", action="store_false", dest="hysteresis", 
                        help="Disable hysteresis (state transition bias)")
    parser.add_argument("--flip-test", action="store_true", help="Average with horizontally flipped image (TTA)")
    return parser.parse_args()

def download_cascade(filename="haarcascade_frontalface_default.xml"):
    """Download Haar cascade if not found"""
    if os.path.exists(filename):
        return filename
    
    print(f"[INFO] Downloading face detector...")
    url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{filename}"
    
    try:
        import urllib.request
        urllib.request.urlretrieve(url, filename)
        print(f"[INFO] Downloaded face detector to {filename}")
        return filename
    except Exception as e:
        print(f"[ERROR] Failed to download: {e}")
        return None

def load_model(ckpt_path, device):
    """Load the FAS model from checkpoint"""
    model = FASViTClassifier(num_classes=2, 
                            num_spoof_types=NUM_SPOOF_TYPES,
                            patch_size=PATCH_SIZE, 
                            depth=DEPTH).to(device)
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()
        print(f"[INFO] Loaded {ckpt_path} (epoch={ckpt.get('epoch','?')})")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

def load_threshold(args):
    """Load threshold from JSON or command line"""
    if args.threshold_json and os.path.exists(args.threshold_json):
        try:
            with open(args.threshold_json, "r") as f:
                data = json.load(f)
                thresh = float(data.get("threshold", args.threshold))
                print(f"[INFO] Using threshold {thresh:.4f} from {args.threshold_json}")
                return thresh
        except Exception as e:
            print(f"[WARN] Failed to load threshold from JSON: {e}")
            
    # Default to higher threshold (0.65) to reduce false spoof detections
    default_thresh = args.threshold if args.threshold != 0.5 else 0.65
    print(f"[INFO] Using threshold {default_thresh:.4f}")
    return default_thresh

def process_frame(frame, face_detector, min_face_size):
    """Detect faces in frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face_size, min_face_size)
    )
    return faces

def expand_face_box(face, margin, frame_shape):
    """Add margin around face box"""
    x, y, w, h = face
    margin_x = int(w * margin)
    margin_y = int(h * margin)
    
    x0 = max(0, x - margin_x)
    y0 = max(0, y - margin_y)
    x1 = min(frame_shape[1], x + w + margin_x)
    y1 = min(frame_shape[0], y + h + margin_y)
    
    return (x0, y0, x1, y1)

def crop_face(frame, box):
    """Crop face region from frame"""
    x0, y0, x1, y1 = box
    return frame[y0:y1, x0:x1]

def check_lighting(face_crop):
    """Analyze lighting conditions of face crop"""
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    return brightness, contrast

def enhance_face(face_crop):
    """Enhance face image to improve detection in various lighting conditions"""
    # Convert to LAB color space
    lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    cl = clahe.apply(l)
    
    # Merge back and convert to BGR
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def predict_spoof_with_tta(model, crop, device):
    """Run model with test-time augmentation (horizontal flip)"""
    # Original image
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb)
    tensor = tfm(img_pil).unsqueeze(0).to(device)
    
    # Flipped image
    flipped = cv2.flip(crop, 1)  # Horizontal flip
    rgb_flip = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    img_flip_pil = Image.fromarray(rgb_flip)
    tensor_flip = tfm(img_flip_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Original prediction
        output = model(tensor)
        if isinstance(output, tuple):
            output = output[0]
        probs = torch.softmax(output, dim=1)[:, 1]
        
        # Flipped prediction
        output_flip = model(tensor_flip)
        if isinstance(output_flip, tuple):
            output_flip = output_flip[0]
        probs_flip = torch.softmax(output_flip, dim=1)[:, 1]
        
        # Average the predictions
        avg_prob = (probs + probs_flip) / 2.0
        
    return avg_prob.item()

def predict_spoof(model, crops, device, use_flip_test=False):
    """Run model on face crops"""
    if not crops:
        return []
    
    if use_flip_test:
        # Process each face individually with TTA
        return [predict_spoof_with_tta(model, crop, device) for crop in crops]
        
    batch = []
    for crop in crops:
        # Convert BGR to RGB and prepare tensor
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)
        tensor = tfm(img_pil)
        batch.append(tensor)
    
    batch_tensor = torch.stack(batch).to(device)
    
    with torch.no_grad():
        output = model(batch_tensor)
        if isinstance(output, tuple):
            output = output[0]  # Get logits from tuple if needed
        probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
        
    return probs

def adjust_threshold(base_threshold, brightness, contrast):
    """Adjust threshold based on lighting conditions"""
    # Very dark scenes tend to get misclassified as spoof
    # Increase threshold in low light (requires more confidence to mark as spoof)
    if brightness < 50:
        # Linear adjustment, more aggressive as light decreases
        # Darker = higher threshold = harder to classify as spoof
        light_factor = max(0, (50 - brightness) / 50 * 0.15)  # Max +0.15 adjustment
        threshold = min(0.95, base_threshold + light_factor)
    # Very bright scenes with low contrast can also be problematic
    elif brightness > 200 and contrast < 30:
        # Bright low-contrast = higher threshold = harder to classify as spoof
        threshold = min(0.95, base_threshold + 0.1)
    else:
        threshold = base_threshold
        
    return threshold

def draw_face_result(frame, face_box, prob, threshold, brightness, contrast):
    """Draw rectangle and label for a face"""
    x0, y0, x1, y1 = face_box
    is_spoof = prob >= threshold
    label = "SPOOF" if is_spoof else "REAL"
    conf = prob if is_spoof else (1.0 - prob)
    color = (0, 0, 255) if is_spoof else (0, 255, 0)  # BGR: Red for spoof, Green for real
    
    # Draw rectangle
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
    
    # Determine confidence level text
    conf_text = ""
    if conf > 0.9:
        conf_text = "HIGH"
    elif conf > 0.7:
        conf_text = "MED"
    else:
        conf_text = "LOW"
    
    # Draw label background
    text = f"{label} {conf:.2f} ({conf_text})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Place text above face when possible
    text_x = x0
    text_y = y0 - 10 if y0 >= 10 else y1 + 20
    
    # Draw text background
    cv2.rectangle(frame, 
                 (text_x, text_y - text_h - 5),
                 (text_x + text_w + 5, text_y + 5), 
                 color, -1)
                 
    # Draw text
    cv2.putText(frame, text, (text_x + 3, text_y - 3), 
                font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Add lighting info if close to threshold
    uncertainty_zone = 0.15
    if abs(prob - threshold) < uncertainty_zone/2:
        light_text = f"Light: {brightness:.0f}, Contrast: {contrast:.0f}"
        cv2.putText(frame, light_text, (text_x + 3, text_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

def main():
    # Parse command line arguments
    args = get_args()
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Load model
    model = load_model(args.ckpt, device)
    if model is None:
        return
    
    # Get detection threshold
    base_threshold = load_threshold(args)
    
    # Load or download face detector
    cascade_path = download_cascade()
    if cascade_path is None:
        print("[ERROR] Could not load face detector")
        return
        
    face_detector = cv2.CascadeClassifier(cascade_path)
    if face_detector.empty():
        print("[ERROR] Failed to initialize face detector")
        return
    
    # Open webcam
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open camera {args.cam}")
        return
    
    # Create save directory if needed
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Tracking variables
    prob_history = {}  # Dictionary to track face probabilities {center_point: [history]}
    state_history = {}  # Dictionary to track face state (spoof/real) {center_point: [states]}
    frame_count = 0
    fps_time = time.time()
    fps = 0
    
    print("[INFO] Starting webcam feed. Press 'q' to quit, 's' to save frame.")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame")
            break
            
        # Count frames for FPS calculation
        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10 / (time.time() - fps_time + 0.001)
            fps_time = time.time()
        
        # Detect faces
        faces = process_frame(frame, face_detector, args.min_face)
        
        # Sort faces by size (area) and limit to max faces
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[:args.max_faces]
            
            # Expand face boxes with margin
            expanded_boxes = [expand_face_box(face, args.margin, frame.shape) for face in faces]
            
            # Crop faces
            crops = [crop_face(frame, box) for box in expanded_boxes]
            
            # Check lighting conditions and enhance if needed
            enhanced_crops = []
            brightness_contrast = []
            
            for crop in crops:
                brightness, contrast = check_lighting(crop)
                brightness_contrast.append((brightness, contrast))
                
                if args.enhance:
                    enhanced_crops.append(enhance_face(crop))
                else:
                    enhanced_crops.append(crop)
            
            # Predict spoofing probabilities
            probs = predict_spoof(model, enhanced_crops, device, use_flip_test=args.flip_test)
            
            # Apply temporal smoothing and state hysteresis
            if args.smooth > 1:
                # Track each face separately
                for i, (x, y, w, h) in enumerate(faces):
                    center = (x + w//2, y + h//2)
                    
                    # Find closest tracked face
                    best_match = None
                    min_dist = float('inf')
                    
                    for tracked_center in list(prob_history.keys()):
                        tx, ty = tracked_center
                        dist = ((tx - center[0])**2 + (ty - center[1])**2)**0.5
                        if dist < min_dist and dist < w/2:  # Within half width
                            min_dist = dist
                            best_match = tracked_center
                    
                    # Get current brightness/contrast
                    brightness, contrast = brightness_contrast[i]
                    
                    # Adjust threshold based on lighting if enabled
                    threshold = base_threshold
                    if args.adaptive_threshold:
                        threshold = adjust_threshold(base_threshold, brightness, contrast)
                    
                    # Current prediction
                    is_spoof = probs[i] >= threshold
                    
                    if best_match is None:
                        # New face
                        prob_history[center] = [probs[i]]
                        state_history[center] = [is_spoof]
                    else:
                        # Update existing face with temporal smoothing and state hysteresis
                        history = prob_history.pop(best_match)
                        states = state_history.pop(best_match)
                        
                        # Add new measurement
                        history.append(probs[i])
                        if len(history) > args.smooth:
                            history.pop(0)
                            
                        # Apply hysteresis for state transitions if enabled
                        if args.hysteresis and len(states) > 0:
                            prev_state = states[-1]
                            # If transitioning from real->spoof, make it harder
                            if not prev_state and is_spoof:
                                # Require 10% more confidence to flip real->spoof
                                hysteresis_adj = 0.1
                                is_spoof = probs[i] >= (threshold + hysteresis_adj)
                        
                        # Add new state
                        states.append(is_spoof)
                        if len(states) > args.smooth:
                            states.pop(0)
                        
                        # Store updated history
                        prob_history[center] = history
                        state_history[center] = states
                        
                        # Weighted average - recent predictions matter more
                        if len(history) > 1:
                            weights = np.linspace(0.6, 1.0, len(history))
                            probs[i] = np.average(history, weights=weights)
                        else:
                            probs[i] = history[0]
                
                # Remove old tracked faces
                if frame_count % 30 == 0:
                    for center in list(prob_history.keys()):
                        if len(prob_history[center]) < args.smooth / 2:
                            del prob_history[center]
                            if center in state_history:
                                del state_history[center]
            
            # Draw results
            for i, box in enumerate(expanded_boxes):
                brightness, contrast = brightness_contrast[i]
                threshold = base_threshold
                if args.adaptive_threshold:
                    threshold = adjust_threshold(base_threshold, brightness, contrast)
                draw_face_result(frame, box, probs[i], threshold, brightness, contrast)
        else:
            # No faces found
            cv2.putText(frame, "No face detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        # Show settings in status bar
        settings = [
            f"FPS: {fps:.1f}",
            f"Faces: {len(faces)}",
            f"Thr: {base_threshold:.2f}",
            f"Enhance: {'ON' if args.enhance else 'OFF'}",
            f"Adapt-Thr: {'ON' if args.adaptive_threshold else 'OFF'}",
            f"Hysteresis: {'ON' if args.hysteresis else 'OFF'}",
            f"q:quit s:save"
        ]
        status_text = "  ".join(settings)
        
        # Draw status bar background
        status_h = 25
        cv2.rectangle(frame, 
                     (0, frame.shape[0] - status_h),
                     (frame.shape[1], frame.shape[0]), 
                     (0, 0, 0), -1)
        
        # Show status text
        cv2.putText(frame, status_text, 
                   (10, frame.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.55, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Display frame
        cv2.imshow("Face Anti-Spoofing", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = int(time.time() * 1000)
            save_path = os.path.join(args.save_dir, f"fas_{timestamp}.jpg")
            cv2.imwrite(save_path, frame)
            print(f"[INFO] Saved frame to {save_path}")
        elif key == ord('e'):
            # Toggle enhancement
            args.enhance = not args.enhance
            print(f"[INFO] Enhancement: {'ON' if args.enhance else 'OFF'}")
        elif key == ord('a'):
            # Toggle adaptive threshold
            args.adaptive_threshold = not args.adaptive_threshold
            print(f"[INFO] Adaptive threshold: {'ON' if args.adaptive_threshold else 'OFF'}")
        elif key == ord('h'):
            # Toggle hysteresis
            args.hysteresis = not args.hysteresis
            print(f"[INFO] Hysteresis: {'ON' if args.hysteresis else 'OFF'}")
        elif key == ord('t'):
            # Increase threshold
            base_threshold = min(0.95, base_threshold + 0.05)
            print(f"[INFO] Threshold: {base_threshold:.2f}")
        elif key == ord('g'):
            # Decrease threshold
            base_threshold = max(0.05, base_threshold - 0.05)
            print(f"[INFO] Threshold: {base_threshold:.2f}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()