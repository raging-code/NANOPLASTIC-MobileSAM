"""
Nanoplastic Detection System – PC Server
Full resolution (1280x720) with optimized MobileSAM for CPU.
Timeout increased to 120 seconds.
"""

import os
import cv2
import csv
import time
import threading
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

WEBCAM_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
JPEG_QUALITY = 85

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

MOCK_FRAME = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
cv2.putText(MOCK_FRAME, "NO CAMERA", (FRAME_WIDTH//2-100, FRAME_HEIGHT//2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

def init_camera():
    if WEBCAM_INDEX < 0:
        return None
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "DEFAULT")]
    for backend, name in backends:
        cap = cv2.VideoCapture(WEBCAM_INDEX, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            print(f"Camera initialized with {name} backend: {FRAME_WIDTH}x{FRAME_HEIGHT}")
            return cap
    return None

def camera_worker():
    global camera, latest_frame, camera_available, camera_error_count
    consecutive_failures = 0
    print("Camera worker started")
    while not stop_camera.is_set():
        if camera is None and camera_error_count < MAX_CAMERA_ERRORS:
            camera = init_camera()
            if camera is None:
                camera_error_count += 1
                if camera_error_count >= MAX_CAMERA_ERRORS:
                    print("Camera unavailable. Using mock frames.")
                    camera_available = False
                    with camera_lock:
                        latest_frame = MOCK_FRAME.copy()
                    frame_ready.set()
                time.sleep(3)
                continue
            else:
                camera_available = True
                camera_error_count = 0
                consecutive_failures = 0

        if camera is None:
            with camera_lock:
                latest_frame = MOCK_FRAME.copy()
            frame_ready.set()
            time.sleep(0.5)
            continue

        ret, frame = camera.read()
        if ret:
            with camera_lock:
                latest_frame = frame.copy()
            frame_ready.set()
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= 5:
                print("Camera read failure, reinitializing...")
                if camera:
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

# ==================== LOAD MOBILESAM (OPTIMIZED FOR CPU, FULL RES) ====================
print("Loading MobileSAM model...")
MOBILESAM_CHECKPOINT = "mobile_sam.pt"
MODEL_TYPE = "vit_t"
if not os.path.exists(MOBILESAM_CHECKPOINT):
    print(f"ERROR: MobileSAM checkpoint not found at {MOBILESAM_CHECKPOINT}")
    exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

sam = sam_model_registry[MODEL_TYPE](checkpoint=MOBILESAM_CHECKPOINT)
sam.to(device=device)

# OPTIMIZED: fewer points, no multi-crop, but full resolution
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=24,               # reduced from 32 for speed
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
    min_mask_region_area=5,
    crop_n_layers=0,                  # disable multi-crop (major speedup)
    crop_n_points_downscale_factor=2,
)
print("MobileSAM loaded with CPU optimizations (full resolution).")

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
    "processing_timeout": 120   # increased for full resolution
}

def process_capture_task():
    global processing_status

    def timeout_handler():
        with processing_lock:
            if processing_status["is_processing"]:
                processing_status["is_processing"] = False
                processing_status["error"] = "Processing timeout (exceeded 120 seconds)"
                print("ERROR: Capture processing timed out")

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

        print(f"[TIMING] Frame captured in {time.time()-start_total:.2f}s")

        particle_count, mean_intensity, avg_confidence, annotated = count_particles_fullres(frame)
        concentration = estimate_concentration(particle_count, mean_intensity)

        risk_level = 'HIGH' if particle_count > 500 else 'MEDIUM' if particle_count > 100 else 'LOW'

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        raw_filename = f"raw_{timestamp}.jpg"
        processed_filename = f"processed_{timestamp}.jpg"
        raw_path = os.path.join(app.config['UPLOAD_FOLDER'], raw_filename)
        processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)

        cv2.imwrite(raw_path, frame)
        cv2.imwrite(processed_path, annotated)

        # CSV with all columns
        csv_file = os.path.join(BASE_DIR, 'detections.csv')
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'particle_count', 'mean_intensity',
                                                   'avg_confidence', 'estimated_concentration_ug_L', 'risk_level'])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'timestamp': timestamp,
                'particle_count': particle_count,
                'mean_intensity': mean_intensity,
                'avg_confidence': avg_confidence,
                'estimated_concentration_ug_L': concentration,
                'risk_level': risk_level
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
            'processed_image': f'/images/processed/{processed_filename}'
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

# ==================== PARTICLE COUNTING (FULL RESOLUTION) ====================
def count_particles_fullres(image_bgr):
    """
    Full-resolution MobileSAM inference with optimized parameters.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    start_inference = time.time()
    masks = mask_generator.generate(image_rgb)
    print(f"[TIMING] MobileSAM inference (full res): {time.time()-start_inference:.2f}s")

    particle_count = 0
    total_intensity = 0.0
    total_confidence = 0.0
    valid_masks = []
    min_area = detection_params['min_particle_area']
    bright_thresh = detection_params['brightness_threshold']

    for mask_dict in masks:
        mask = mask_dict["segmentation"]
        area = np.sum(mask)
        if area < min_area:
            continue
        mean_intensity = np.mean(gray[mask])
        if mean_intensity < bright_thresh:
            continue
        particle_count += 1
        total_intensity += mean_intensity
        
        # --- FIX: use stability_score instead of predicted_iou (which may be 0) ---
        # Try predicted_iou first, then stability_score, then default 0.5
        confidence = mask_dict.get("predicted_iou")
        if confidence is None:
            confidence = mask_dict.get("stability_score", 0.5)
        confidence = float(confidence)
        total_confidence += confidence
        valid_masks.append(mask)

        # Optional debug (remove in production)
        # print(f"Mask confidence: {confidence:.3f}")

    avg_intensity = total_intensity / particle_count if particle_count > 0 else 0.0
    avg_confidence = total_confidence / particle_count if particle_count > 0 else 0.0

    # Annotate
    annotated = image_bgr.copy()
    for mask in valid_masks:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)

    return particle_count, round(avg_intensity, 3), round(avg_confidence, 3), annotated

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
    return jsonify({'available': camera_available})

@app.route('/capture', methods=['POST', 'GET'])
def capture_and_analyze():
    """Synchronous capture (for testing)."""
    try:
        frame = get_latest_frame()
        if frame is None:
            return jsonify({'status': 'error', 'error': 'No frame available'}), 500

        particle_count, mean_intensity, avg_confidence, annotated = count_particles_fullres(frame)
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
                                                   'avg_confidence', 'estimated_concentration_ug_L', 'risk_level'])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'timestamp': timestamp,
                'particle_count': particle_count,
                'mean_intensity': mean_intensity,
                'avg_confidence': avg_confidence,
                'estimated_concentration_ug_L': concentration,
                'risk_level': risk_level
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
            'processed_image': f'/images/processed/processed_{timestamp}.jpg'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/trigger_capture', methods=['GET', 'POST'])
def trigger_capture_async():
    global processing_status
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
    global processing_status
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
            # Handle old CSV files gracefully
            clean_row = {
                'timestamp': row.get('timestamp', ''),
                'particle_count': row.get('particle_count', '0'),
                'mean_intensity': row.get('mean_intensity', '0'),
                'avg_confidence': row.get('avg_confidence', '0'),
                'estimated_concentration_ug_L': row.get('estimated_concentration_ug_L', '0'),
                'risk_level': row.get('risk_level', 'LOW')
            }
            # Replace None values
            for k, v in clean_row.items():
                if v is None:
                    clean_row[k] = '' if k == 'timestamp' else '0'
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

@app.route('/api/latest_image')
def latest_image():
    processed_dir = PROCESSED_FOLDER
    files = [f for f in os.listdir(processed_dir) if f.startswith('processed_') and f.endswith('.jpg')]
    if not files:
        return jsonify({'filename': None})
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(processed_dir, f)))
    return jsonify({'filename': latest_file})

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    print("=" * 50)
    print("Nanoplastic Detection – Full Resolution + Optimized MobileSAM")
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