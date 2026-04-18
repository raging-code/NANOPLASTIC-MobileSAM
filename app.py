"""
Nanoplastic Detection System – PC Server
Full resolution (1280x720) with combined MobileSAM + Blob Detector.
Supports USB webcam selection and multiple detection modes.
"""

import os
import cv2
import csv
import time
import threading
import json
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from waitress import serve

from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
import torch

# ==================== CONFIGURATION ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'images', 'received')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'images', 'processed')
CONFIG_FOLDER = os.path.join(BASE_DIR, 'config')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(CONFIG_FOLDER, exist_ok=True)

# Camera configuration
CAMERA_INDEX = int(os.environ.get('CAMERA_INDEX', 0))
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
JPEG_QUALITY = 85

# Detection mode: 'blob', 'sam', or 'combined'
DETECTION_MODE = os.environ.get('DETECTION_MODE', 'combined')

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==================== BACKGROUND FRAME GRABBER ====================
camera = None
camera_lock = threading.Lock()
latest_frame = None
frame_ready = threading.Event()
stop_camera = threading.Event()
camera_available = False
camera_error_count = 0
MAX_CAMERA_ERRORS = 3

def find_working_camera(max_index=5):
    print("Scanning for available cameras...")
    working_indices = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  Camera index {i}: OK (resolution {w}x{h})")
                working_indices.append(i)
            cap.release()
        else:
            print(f"  Camera index {i}: not available")
    if working_indices:
        if CAMERA_INDEX in working_indices:
            return CAMERA_INDEX
        if 0 in working_indices and len(working_indices) > 1:
            print("  Built-in camera (index 0) detected. Suggest using USB webcam at index", working_indices[1])
            return working_indices[1]
        return working_indices[0]
    return None

def init_camera():
    camera_idx = find_working_camera()
    if camera_idx is None:
        return None
    print(f"Opening camera index {camera_idx} with resolution {FRAME_WIDTH}x{FRAME_HEIGHT}")
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera initialized: requested {FRAME_WIDTH}x{FRAME_HEIGHT}, got {actual_width}x{actual_height}")
        return cap
    return None

def camera_worker():
    global camera, latest_frame, camera_available, camera_error_count
    consecutive_failures = 0
    while not stop_camera.is_set():
        if camera is None and camera_error_count < MAX_CAMERA_ERRORS:
            camera = init_camera()
            if camera is None:
                camera_error_count += 1
                if camera_error_count >= MAX_CAMERA_ERRORS:
                    print("Camera unavailable. Using mock frames.")
                    camera_available = False
                    mock = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                    cv2.putText(mock, "NO CAMERA", (FRAME_WIDTH//2-100, FRAME_HEIGHT//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    with camera_lock:
                        latest_frame = mock
                    frame_ready.set()
                time.sleep(3)
                continue
            else:
                camera_available = True
                camera_error_count = 0
                consecutive_failures = 0

        if camera is None:
            mock = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            cv2.putText(mock, "NO CAMERA", (FRAME_WIDTH//2-100, FRAME_HEIGHT//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            with camera_lock:
                latest_frame = mock
            frame_ready.set()
            time.sleep(0.5)
            continue

        ret, frame = camera.read()
        if ret:
            if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            with camera_lock:
                latest_frame = frame.copy()
            frame_ready.set()
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= 5:
                print("Camera read failure, reinitializing...")
                camera.release()
                camera = None
                time.sleep(2)
            else:
                time.sleep(0.5)
        time.sleep(0.03)

    if camera:
        camera.release()
    print("Camera worker stopped")

def get_latest_frame(timeout=5.0):
    if frame_ready.wait(timeout=timeout):
        with camera_lock:
            if latest_frame is not None:
                return latest_frame.copy()
    return None

camera_thread = threading.Thread(target=camera_worker, daemon=True)
camera_thread.start()
time.sleep(3)

# ==================== LOAD MOBILESAM (if needed) ====================
sam = None
mask_generator = None
if DETECTION_MODE in ['sam', 'combined']:
    print("Loading MobileSAM model...")
    MOBILESAM_CHECKPOINT = "mobile_sam.pt"
    MODEL_TYPE = "vit_t"
    if not os.path.exists(MOBILESAM_CHECKPOINT):
        print(f"ERROR: MobileSAM checkpoint not found at {MOBILESAM_CHECKPOINT}")
        print("Falling back to blob-only detection.")
        DETECTION_MODE = 'blob'
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        sam = sam_model_registry[MODEL_TYPE](checkpoint=MOBILESAM_CHECKPOINT)
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=24,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            min_mask_region_area=5,
            crop_n_layers=0,
            crop_n_points_downscale_factor=2,
        )
        print("MobileSAM loaded.")
else:
    print("MobileSAM disabled (blob-only mode).")

# ==================== BLOB DETECTOR PARAMETERS ====================
# Tune these for your optical setup
blob_params = {
    "minArea": 3,          # pixels – small nanoplastics might be 2-5 pixels
    "maxArea": 200,        # exclude large dust
    "minCircularity": 0.3, # allow irregular shapes
    "minConvexity": 0.5,
    "minInertiaRatio": 0.2,
    "thresholdStep": 5,
    "minThreshold": 10,
    "maxThreshold": 200,
    "repeatability": 2,
}

def create_blob_detector(params):
    """Create OpenCV SimpleBlobDetector with given parameters."""
    detector_params = cv2.SimpleBlobDetector_Params()
    detector_params.filterByArea = True
    detector_params.minArea = params["minArea"]
    detector_params.maxArea = params["maxArea"]
    detector_params.filterByCircularity = True
    detector_params.minCircularity = params["minCircularity"]
    detector_params.filterByConvexity = True
    detector_params.minConvexity = params["minConvexity"]
    detector_params.filterByInertia = True
    detector_params.minInertiaRatio = params["minInertiaRatio"]
    detector_params.thresholdStep = params["thresholdStep"]
    detector_params.minThreshold = params["minThreshold"]
    detector_params.maxThreshold = params["maxThreshold"]
    detector_params.minRepeatability = params["repeatability"]
    return cv2.SimpleBlobDetector_create(detector_params)

blob_detector = create_blob_detector(blob_params)

# ==================== CALIBRATION ====================
calibration = {}
calibration_file = os.path.join(CONFIG_FOLDER, 'calibration_curve.csv')
if os.path.exists(calibration_file):
    with open(calibration_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            calibration[float(row['concentration_ug_L'])] = {
                'particle_count': int(row['particle_count']),
                'intensity': float(row['intensity'])
            }
else:
    calibration = {
        0: {'particle_count': 5, 'intensity': 0.02},
        1: {'particle_count': 450, 'intensity': 0.25},
        5: {'particle_count': 2200, 'intensity': 0.68},
        10: {'particle_count': 4700, 'intensity': 0.92},
        20: {'particle_count': 10500, 'intensity': 1.00}
    }

detection_params = {
    "min_particle_area": 5,
    "brightness_threshold": 30,
    "max_particles_per_frame": 5000,
}

# ==================== ASYNC CAPTURE STATE ====================
processing_lock = threading.Lock()
processing_status = {
    "is_processing": False,
    "last_result": None,
    "error": None,
    "start_time": None,
    "processing_timeout": 120
}

# ==================== PARTICLE DETECTION FUNCTIONS ====================
def detect_blobs(image_gray, draw_on_image=None):
    """
    Detect bright blobs (small particles) using OpenCV's blob detector.
    Returns: list of (x, y, size, confidence) and annotated image (if draw_on_image provided).
    """
    # Blob detector works best on inverted image? No, it finds dark blobs by default.
    # We want bright spots on dark background -> invert.
    # Alternatively, use SimpleBlobDetector with parameters that detect bright blobs.
    # By default, it thresholds from minThreshold to maxThreshold and finds dark regions.
    # To find bright blobs, we can invert the image or set 'blobColor' to 255.
    # However, SimpleBlobDetector_Params does not have a direct 'blobColor' in OpenCV.
    # So we invert the image: dark becomes bright.
    inverted = 255 - image_gray
    keypoints = blob_detector.detect(inverted)
    
    detections = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        size = kp.size  # diameter in pixels
        # Use response as confidence (if available)
        confidence = kp.response if hasattr(kp, 'response') and kp.response > 0 else 0.8
        detections.append((x, y, size, confidence))
    
    annotated = None
    if draw_on_image is not None:
        annotated = draw_on_image.copy()
        for (x, y, size, _) in detections:
            radius = int(size / 2)
            cv2.circle(annotated, (x, y), radius, (255, 0, 0), 2)  # blue circles for blobs
            cv2.circle(annotated, (x, y), 2, (0, 255, 255), -1)
    return detections, annotated

def detect_sam(image_rgb, gray, draw_on_image=None):
    """Run MobileSAM and return masks filtered by area and brightness."""
    if mask_generator is None:
        return [], None
    
    masks = mask_generator.generate(image_rgb)
    valid_masks = []
    for mask_dict in masks:
        mask = mask_dict["segmentation"]
        area = np.sum(mask)
        if area < detection_params['min_particle_area'] or area > 5000:
            continue
        mean_intensity = np.mean(gray[mask])
        if mean_intensity < detection_params['brightness_threshold']:
            continue
        valid_masks.append(mask_dict)
    
    annotated = None
    if draw_on_image is not None:
        annotated = draw_on_image.copy()
        for mask_dict in valid_masks:
            mask = mask_dict["segmentation"]
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)  # green for SAM
    return valid_masks, annotated

def count_particles_combined(image_bgr):
    """
    Combine blob detection (for tiny particles) and SAM (for larger objects).
    Returns total count, average intensity, average confidence, annotated image.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    total_count = 0
    total_intensity = 0.0
    total_confidence = 0.0
    
    # Start with a base annotated image
    annotated = image_bgr.copy()
    
    # 1. Blob detection for small bright spots
    blob_dets, blob_annotated = detect_blobs(gray, draw_on_image=annotated)
    if blob_annotated is not None:
        annotated = blob_annotated
    for (x, y, size, conf) in blob_dets:
        # Estimate intensity at blob center
        radius = int(size/2)
        y1, y2 = max(0, y-radius), min(gray.shape[0], y+radius)
        x1, x2 = max(0, x-radius), min(gray.shape[1], x+radius)
        if y2 > y1 and x2 > x1:
            roi = gray[y1:y2, x1:x2]
            intensity = np.mean(roi) if roi.size > 0 else 0
        else:
            intensity = gray[y, x] if 0 <= y < gray.shape[0] and 0 <= x < gray.shape[1] else 0
        total_intensity += intensity
        total_confidence += conf
        total_count += 1
    
    # 2. SAM detection for larger objects (optional)
    if DETECTION_MODE in ['sam', 'combined'] and mask_generator is not None:
        sam_masks, sam_annotated = detect_sam(image_rgb, gray, draw_on_image=annotated)
        if sam_annotated is not None:
            annotated = sam_annotated
        for mask_dict in sam_masks:
            mask = mask_dict["segmentation"]
            area = np.sum(mask)
            # Avoid double-counting if a blob is also inside a SAM mask?
            # For simplicity, we count SAM masks as separate particles.
            # You could add overlap checking, but not necessary for now.
            mean_intensity = np.mean(gray[mask])
            confidence = mask_dict.get("predicted_iou", mask_dict.get("stability_score", 0.5))
            total_intensity += mean_intensity
            total_confidence += confidence
            total_count += 1
    
    # Prevent division by zero
    if total_count == 0:
        avg_intensity = 0.0
        avg_confidence = 0.0
    else:
        avg_intensity = total_intensity / total_count
        avg_confidence = total_confidence / total_count
    
    # Cap at max_particles_per_frame
    max_particles = detection_params.get('max_particles_per_frame', 5000)
    if total_count > max_particles:
        total_count = max_particles
    
    return total_count, round(avg_intensity, 3), round(avg_confidence, 3), annotated

def count_particles_blob_only(image_bgr):
    """Only blob detection (faster, no SAM)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    detections, annotated = detect_blobs(gray, draw_on_image=image_bgr)
    total_count = len(detections)
    total_intensity = 0.0
    total_confidence = 0.0
    for (x, y, size, conf) in detections:
        # Compute intensity at blob center
        radius = max(1, int(size/2))
        y1, y2 = max(0, y-radius), min(gray.shape[0], y+radius)
        x1, x2 = max(0, x-radius), min(gray.shape[1], x+radius)
        roi = gray[y1:y2, x1:x2]
        intensity = np.mean(roi) if roi.size > 0 else 0
        total_intensity += intensity
        total_confidence += conf
    if total_count > 0:
        avg_intensity = total_intensity / total_count
        avg_confidence = total_confidence / total_count
    else:
        avg_intensity = 0.0
        avg_confidence = 0.0
    return total_count, round(avg_intensity, 3), round(avg_confidence, 3), annotated

def count_particles_sam_only(image_bgr):
    """Only MobileSAM (original behavior)."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if mask_generator is None:
        return 0, 0.0, 0.0, image_bgr
    masks, annotated = detect_sam(image_rgb, gray, draw_on_image=image_bgr)
    total_count = len(masks)
    total_intensity = 0.0
    total_confidence = 0.0
    for mask_dict in masks:
        mask = mask_dict["segmentation"]
        mean_intensity = np.mean(gray[mask])
        confidence = mask_dict.get("predicted_iou", mask_dict.get("stability_score", 0.5))
        total_intensity += mean_intensity
        total_confidence += confidence
    if total_count > 0:
        avg_intensity = total_intensity / total_count
        avg_confidence = total_confidence / total_count
    else:
        avg_intensity = 0.0
        avg_confidence = 0.0
    return total_count, round(avg_intensity, 3), round(avg_confidence, 3), annotated

def estimate_concentration(particle_count, intensity):
    concentrations = sorted(calibration.keys())
    if particle_count <= calibration[concentrations[0]]['particle_count']:
        return concentrations[0]
    if particle_count >= calibration[concentrations[-1]]['particle_count']:
        return concentrations[-1]
    for i in range(len(concentrations)-1):
        c_low, c_high = concentrations[i], concentrations[i+1]
        pc_low = calibration[c_low]['particle_count']
        pc_high = calibration[c_high]['particle_count']
        if pc_low <= particle_count <= pc_high:
            ratio = (particle_count - pc_low) / (pc_high - pc_low)
            return round(c_low + ratio * (c_high - c_low), 2)
    return concentrations[0]

def process_capture_task():
    global processing_status
    def timeout_handler():
        with processing_lock:
            if processing_status["is_processing"]:
                processing_status["is_processing"] = False
                processing_status["error"] = "Processing timeout (exceeded 120 seconds)"
    timer = threading.Timer(processing_status["processing_timeout"], timeout_handler)
    timer.start()
    try:
        start_total = time.time()
        frame = get_latest_frame(timeout=10.0)
        if frame is None:
            with processing_lock:
                processing_status["is_processing"] = False
                processing_status["error"] = "No frame available from camera"
            timer.cancel()
            return
        
        # Choose detection method
        if DETECTION_MODE == 'blob':
            particle_count, mean_intensity, avg_confidence, annotated = count_particles_blob_only(frame)
        elif DETECTION_MODE == 'sam':
            particle_count, mean_intensity, avg_confidence, annotated = count_particles_sam_only(frame)
        else:  # 'combined'
            particle_count, mean_intensity, avg_confidence, annotated = count_particles_combined(frame)
        
        concentration = estimate_concentration(particle_count, mean_intensity)
        risk_level = 'HIGH' if particle_count > 500 else 'MEDIUM' if particle_count > 100 else 'LOW'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        raw_filename = f"raw_{timestamp}.jpg"
        processed_filename = f"processed_{timestamp}.jpg"
        raw_path = os.path.join(app.config['UPLOAD_FOLDER'], raw_filename)
        processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
        
        cv2.imwrite(raw_path, frame)
        cv2.imwrite(processed_path, annotated)
        
        csv_file = os.path.join(BASE_DIR, 'detections.csv')
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'particle_count', 'mean_intensity',
                                                   'avg_confidence', 'estimated_concentration_ug_L', 'risk_level',
                                                   'detection_mode'])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'timestamp': timestamp,
                'particle_count': particle_count,
                'mean_intensity': mean_intensity,
                'avg_confidence': avg_confidence,
                'estimated_concentration_ug_L': concentration,
                'risk_level': risk_level,
                'detection_mode': DETECTION_MODE
            })
        
        result = {
            'status': 'ok',
            'timestamp': timestamp,
            'particle_count': particle_count,
            'mean_intensity': mean_intensity,
            'avg_confidence': avg_confidence,
            'estimated_concentration_ug_L': concentration,
            'risk_level': risk_level,
            'raw_image': f'/images/received/{raw_filename}',
            'processed_image': f'/images/processed/{processed_filename}',
            'detection_mode': DETECTION_MODE
        }
        with processing_lock:
            processing_status["is_processing"] = False
            processing_status["last_result"] = result
            processing_status["error"] = None
        print(f"[TIMING] Total processing time: {time.time()-start_total:.2f}s")
    except Exception as e:
        import traceback
        traceback.print_exc()
        with processing_lock:
            processing_status["is_processing"] = False
            processing_status["error"] = str(e)
    finally:
        timer.cancel()

# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'alive', 'timestamp': datetime.now().isoformat()})

@app.route('/images/processed/<filename>')
def serve_processed(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route('/images/received/<filename>')
def serve_received(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/api/camera_status')
def camera_status():
    return jsonify({'available': camera_available, 'camera_index': CAMERA_INDEX})

@app.route('/capture', methods=['POST', 'GET'])
def capture_and_analyze():
    """Synchronous capture (for testing)."""
    try:
        frame = get_latest_frame()
        if frame is None:
            return jsonify({'status': 'error', 'error': 'No frame available'}), 500
        
        if DETECTION_MODE == 'blob':
            particle_count, mean_intensity, avg_confidence, annotated = count_particles_blob_only(frame)
        elif DETECTION_MODE == 'sam':
            particle_count, mean_intensity, avg_confidence, annotated = count_particles_sam_only(frame)
        else:
            particle_count, mean_intensity, avg_confidence, annotated = count_particles_combined(frame)
        
        concentration = estimate_concentration(particle_count, mean_intensity)
        risk_level = 'HIGH' if particle_count > 500 else 'MEDIUM' if particle_count > 100 else 'LOW'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        raw_path = os.path.join(app.config['UPLOAD_FOLDER'], f"raw_{timestamp}.jpg")
        proc_path = os.path.join(PROCESSED_FOLDER, f"processed_{timestamp}.jpg")
        cv2.imwrite(raw_path, frame)
        cv2.imwrite(proc_path, annotated)
        
        csv_file = os.path.join(BASE_DIR, 'detections.csv')
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'particle_count', 'mean_intensity',
                                                   'avg_confidence', 'estimated_concentration_ug_L', 'risk_level',
                                                   'detection_mode'])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'timestamp': timestamp,
                'particle_count': particle_count,
                'mean_intensity': mean_intensity,
                'avg_confidence': avg_confidence,
                'estimated_concentration_ug_L': concentration,
                'risk_level': risk_level,
                'detection_mode': DETECTION_MODE
            })
        
        return jsonify({
            'status': 'ok',
            'timestamp': timestamp,
            'particle_count': particle_count,
            'mean_intensity': mean_intensity,
            'avg_confidence': avg_confidence,
            'estimated_concentration_ug_L': concentration,
            'risk_level': risk_level,
            'raw_image': f'/images/received/raw_{timestamp}.jpg',
            'processed_image': f'/images/processed/processed_{timestamp}.jpg',
            'detection_mode': DETECTION_MODE
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/trigger_capture', methods=['GET', 'POST'])
def trigger_capture_async():
    with processing_lock:
        if processing_status["is_processing"]:
            return jsonify({'status': 'error', 'error': 'Capture already in progress'}), 409
        processing_status["is_processing"] = True
        processing_status["error"] = None
        processing_status["start_time"] = time.time()
        processing_status["last_result"] = None
    thread = threading.Thread(target=process_capture_task, daemon=True)
    thread.start()
    return jsonify({'status': 'accepted', 'message': 'Capture started'}), 202

@app.route('/api/capture_status', methods=['GET'])
def get_capture_status():
    with processing_lock:
        return jsonify({
            'is_processing': processing_status["is_processing"],
            'last_result': processing_status["last_result"],
            'error': processing_status["error"]
        })

@app.route('/api/reset_processing', methods=['POST'])
def reset_processing():
    with processing_lock:
        was_processing = processing_status["is_processing"]
        processing_status["is_processing"] = False
        processing_status["error"] = "Manually reset"
        processing_status["last_result"] = None
    return jsonify({'status': 'ok', 'was_processing': was_processing})

@app.route('/api/detections')
def get_detections():
    csv_file = os.path.join(BASE_DIR, 'detections.csv')
    if not os.path.exists(csv_file):
        return jsonify([])
    detections = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = {
                'timestamp': row.get('timestamp', ''),
                'particle_count': row.get('particle_count', '0'),
                'mean_intensity': row.get('mean_intensity', '0'),
                'avg_confidence': row.get('avg_confidence', '0'),
                'estimated_concentration_ug_L': row.get('estimated_concentration_ug_L', '0'),
                'risk_level': row.get('risk_level', 'LOW'),
                'detection_mode': row.get('detection_mode', 'unknown')
            }
            detections.append(clean_row)
    return jsonify(detections[-50:])

@app.route('/api/stats')
def get_stats():
    csv_file = os.path.join(BASE_DIR, 'detections.csv')
    if not os.path.exists(csv_file):
        return jsonify({'total_samples': 0, 'avg_particles': 0, 'high_risk_count': 0,
                        'medium_risk_count': 0, 'low_risk_count': 0, 'max_particles': 0})
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        detections = list(reader)
    if not detections:
        return jsonify({'total_samples': 0, 'avg_particles': 0, 'high_risk_count': 0,
                        'medium_risk_count': 0, 'low_risk_count': 0, 'max_particles': 0})
    total = len(detections)
    counts = []
    for d in detections:
        try:
            counts.append(int(d.get('particle_count', 0)))
        except:
            counts.append(0)
    avg_particles = sum(counts) / total if total else 0
    high = sum(1 for d in detections if d.get('risk_level') == 'HIGH')
    medium = sum(1 for d in detections if d.get('risk_level') == 'MEDIUM')
    low = sum(1 for d in detections if d.get('risk_level') == 'LOW')
    return jsonify({
        'total_samples': total,
        'avg_particles': round(avg_particles, 1),
        'high_risk_count': high,
        'medium_risk_count': medium,
        'low_risk_count': low,
        'max_particles': max(counts) if counts else 0
    })

@app.route('/api/set_detection_mode', methods=['POST'])
def set_detection_mode():
    """Change detection mode at runtime (blob, sam, combined)."""
    global DETECTION_MODE
    data = request.get_json()
    mode = data.get('mode', '').lower()
    if mode not in ['blob', 'sam', 'combined']:
        return jsonify({'status': 'error', 'error': 'Invalid mode. Choose blob, sam, or combined'}), 400
    DETECTION_MODE = mode
    return jsonify({'status': 'ok', 'detection_mode': DETECTION_MODE})

@app.route('/api/blob_params', methods=['GET', 'POST'])
def blob_params_route():
    global blob_detector, blob_params
    if request.method == 'GET':
        return jsonify(blob_params)
    else:
        new_params = request.get_json()
        blob_params.update(new_params)
        blob_detector = create_blob_detector(blob_params)
        return jsonify({'status': 'ok', 'params': blob_params})

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    print("=" * 50)
    print("Nanoplastic Detection – Combined MobileSAM + Blob Detector")
    print(f"Detection mode: {DETECTION_MODE}")
    print("Timeout: 120 seconds")
    print("=" * 50)
    print(f"Dashboard: http://localhost:5000/")
    print(f"Async Capture: http://localhost:5000/trigger_capture")
    print(f"Status: http://localhost:5000/api/capture_status")
    print("=" * 50)
    try:
        serve(app, host='0.0.0.0', port=5000, threads=4)
    finally:
        stop_camera.set()
        camera_thread.join(timeout=2)