import os
import json
import csv
import time
import threading
import cv2
import numpy as np
import torch
import requests
from datetime import datetime
from flask import Flask, Response, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from waitress import serve

# ---------- Nanoplastic imports ----------
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

# ==================== CONFIGURATION ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Raspberry Pi USV server address (must be reachable)
RASPI_URL = os.environ.get("RASPI_URL", "http://192.168.0.108:8000/")

# Nanoplastic configuration
CAMERA_INDEX = int(os.environ.get('CAMERA_INDEX', 0))
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
DETECTION_MODE = os.environ.get('DETECTION_MODE', 'combined')
MOBILESAM_CHECKPOINT = "mobile_sam.pt"
MODEL_TYPE = "vit_t"

# Directories for nanoplastic images
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'images', 'received')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'images', 'processed')
CONFIG_FOLDER = os.path.join(BASE_DIR, 'config')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(CONFIG_FOLDER, exist_ok=True)

# ==================== FLASK APP ====================
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==================== NANOPLASTIC DETECTION GLOBALS ====================
camera = None
camera_lock = threading.Lock()
latest_frame = None
frame_ready = threading.Event()
stop_camera = threading.Event()
camera_available = False
camera_error_count = 0
MAX_CAMERA_ERRORS = 3

sam = None
mask_generator = None
if DETECTION_MODE in ['sam', 'combined']:
    print("Loading MobileSAM...")
    if not os.path.exists(MOBILESAM_CHECKPOINT):
        print(f"ERROR: {MOBILESAM_CHECKPOINT} not found. Falling back to blob-only.")
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

blob_params = {
    "minArea": 3, "maxArea": 200, "minCircularity": 0.3, "minConvexity": 0.5,
    "minInertiaRatio": 0.2, "thresholdStep": 5, "minThreshold": 10,
    "maxThreshold": 200, "repeatability": 2,
}

def create_blob_detector(params):
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

processing_lock = threading.Lock()
processing_status = {
    "is_processing": False,
    "last_result": None,
    "error": None,
    "start_time": None,
    "processing_timeout": 120
}

# ---------- Nanoplastic helper functions (same as before) ----------
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

def detect_blobs(image_gray, draw_on_image=None):
    inverted = 255 - image_gray
    keypoints = blob_detector.detect(inverted)
    detections = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        size = kp.size
        confidence = kp.response if hasattr(kp, 'response') and kp.response > 0 else 0.8
        detections.append((x, y, size, confidence))
    annotated = None
    if draw_on_image is not None:
        annotated = draw_on_image.copy()
        for (x, y, size, _) in detections:
            radius = int(size / 2)
            cv2.circle(annotated, (x, y), radius, (255, 0, 0), 2)
            cv2.circle(annotated, (x, y), 2, (0, 255, 255), -1)
    return detections, annotated

def detect_sam(image_rgb, gray, draw_on_image=None):
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
            cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)
    return valid_masks, annotated

def count_particles_combined(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    total_count = 0
    total_intensity = 0.0
    total_confidence = 0.0
    annotated = image_bgr.copy()
    blob_dets, blob_annotated = detect_blobs(gray, draw_on_image=annotated)
    if blob_annotated is not None:
        annotated = blob_annotated
    for (x, y, size, conf) in blob_dets:
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
    if DETECTION_MODE in ['sam', 'combined'] and mask_generator is not None:
        sam_masks, sam_annotated = detect_sam(image_rgb, gray, draw_on_image=annotated)
        if sam_annotated is not None:
            annotated = sam_annotated
        for mask_dict in sam_masks:
            mask = mask_dict["segmentation"]
            mean_intensity = np.mean(gray[mask])
            confidence = mask_dict.get("predicted_iou", mask_dict.get("stability_score", 0.5))
            total_intensity += mean_intensity
            total_confidence += confidence
            total_count += 1
    if total_count == 0:
        avg_intensity = 0.0
        avg_confidence = 0.0
    else:
        avg_intensity = total_intensity / total_count
        avg_confidence = total_confidence / total_count
    max_particles = detection_params.get('max_particles_per_frame', 5000)
    if total_count > max_particles:
        total_count = max_particles
    return total_count, round(avg_intensity, 3), round(avg_confidence, 3), annotated

def count_particles_blob_only(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    detections, annotated = detect_blobs(gray, draw_on_image=image_bgr)
    total_count = len(detections)
    total_intensity = 0.0
    total_confidence = 0.0
    for (x, y, size, conf) in detections:
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
        if DETECTION_MODE == 'blob':
            particle_count, mean_intensity, avg_confidence, annotated = count_particles_blob_only(frame)
        elif DETECTION_MODE == 'sam':
            particle_count, mean_intensity, avg_confidence, annotated = count_particles_sam_only(frame)
        else:
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

# ==================== PROXY FUNCTIONS TO RASPBERRY PI ====================
def proxy_to_pi(endpoint, method='GET', data=None, params=None):
    url = f"{RASPI_URL}{endpoint}"
    try:
        if method == 'GET':
            resp = requests.get(url, params=params, timeout=5)
        elif method == 'POST':
            resp = requests.post(url, json=data, timeout=5)
        else:
            return None, 405
        return resp.content, resp.status_code, resp.headers.get('Content-Type', 'text/plain')
    except Exception as e:
        return str(e).encode(), 503, 'text/plain'

# ==================== HEAVY METAL LOGGING (every 5 minutes) ====================
latest_telemetry = None

@app.route('/data')
def proxy_data():
    content, status, ctype = proxy_to_pi('/data')
    # Cache latest telemetry for heavy metal logging
    if status == 200 and ctype == 'application/json':
        try:
            data = json.loads(content.decode('utf-8'))
            global latest_telemetry
            latest_telemetry = data
        except:
            pass
    return Response(content, status=status, mimetype=ctype)

def heavy_metal_logger():
    while True:
        time.sleep(300)  # 5 minutes
        if latest_telemetry:
            water_temp = latest_telemetry.get('waterTemp', '')
            water_ph = latest_telemetry.get('waterPH', '')
            water_tds = latest_telemetry.get('waterTDS', '')
            risk_level = latest_telemetry.get('hmRiskLevel', '')
            timestamp = datetime.now()
            date_str = timestamp.strftime('%Y-%m-%d')
            time_str = timestamp.strftime('%H:%M:%S')
            csv_file = os.path.join(BASE_DIR, 'heavy_metal_log.csv')
            file_exists = os.path.isfile(csv_file)
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['Date', 'Time', 'TDS (ppm)', 'pH', 'Temp (°C)', 'Risk'])
                writer.writerow([date_str, time_str, water_tds, water_ph, water_temp, risk_level])

# ==================== NEW ENDPOINTS ====================
@app.route('/api/latest_image')
def api_latest_image():
    """Return the filename of the most recent processed image."""
    if not os.path.exists(PROCESSED_FOLDER):
        return jsonify({'filename': None})
    files = [f for f in os.listdir(PROCESSED_FOLDER) if f.startswith('processed_') and f.endswith('.jpg')]
    if not files:
        return jsonify({'filename': None})
    latest = sorted(files)[-1]
    return jsonify({'filename': latest})

@app.route('/api/heavy_metal_log')
def api_heavy_metal_log():
    """Return the contents of heavy_metal_log.csv as JSON."""
    csv_file = os.path.join(BASE_DIR, 'heavy_metal_log.csv')
    if not os.path.exists(csv_file):
        return jsonify([])
    data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'Date': row.get('Date', ''),
                'Time': row.get('Time', ''),
                'TDS (ppm)': row.get('TDS (ppm)', ''),
                'pH': row.get('pH', ''),
                'Temp (°C)': row.get('Temp (°C)', ''),
                'Risk': row.get('Risk', '')
            })
    # Return last 100 entries
    return jsonify(data[-100:])

# ==================== FLASK ROUTES (unchanged) ====================
@app.route('/')
def index():
    return render_template('index.html')

# ---------- Proxy routes for USV ----------
@app.route('/path')
def proxy_path():
    content, status, ctype = proxy_to_pi('/path')
    return Response(content, status=status, mimetype=ctype)

@app.route('/addWP')
def proxy_add_wp():
    params = request.args.to_dict()
    content, status, ctype = proxy_to_pi('/addWP', params=params)
    return Response(content, status=status, mimetype=ctype)

@app.route('/start')
def proxy_start():
    content, status, ctype = proxy_to_pi('/start')
    return Response(content, status=status, mimetype=ctype)

@app.route('/stop')
def proxy_stop():
    content, status, ctype = proxy_to_pi('/stop')
    return Response(content, status=status, mimetype=ctype)

@app.route('/clear')
def proxy_clear():
    content, status, ctype = proxy_to_pi('/clear')
    return Response(content, status=status, mimetype=ctype)

@app.route('/calibrate')
def proxy_calibrate():
    content, status, ctype = proxy_to_pi('/calibrate')
    return Response(content, status=status, mimetype=ctype)

@app.route('/delWP')
def proxy_del_wp():
    params = request.args.to_dict()
    content, status, ctype = proxy_to_pi('/delWP', params=params)
    return Response(content, status=status, mimetype=ctype)

@app.route('/setWP')
def proxy_set_wp():
    params = request.args.to_dict()
    content, status, ctype = proxy_to_pi('/setWP', params=params)
    return Response(content, status=status, mimetype=ctype)

@app.route('/servo')
def proxy_servo():
    params = request.args.to_dict()
    content, status, ctype = proxy_to_pi('/servo', params=params)
    return Response(content, status=status, mimetype=ctype)

@app.route('/pump')
def proxy_pump():
    params = request.args.to_dict()
    content, status, ctype = proxy_to_pi('/pump', params=params)
    return Response(content, status=status, mimetype=ctype)

@app.route('/set-left')
def proxy_set_left():
    params = request.args.to_dict()
    content, status, ctype = proxy_to_pi('/set-left', params=params)
    return Response(content, status=status, mimetype=ctype)

@app.route('/set-right')
def proxy_set_right():
    params = request.args.to_dict()
    content, status, ctype = proxy_to_pi('/set-right', params=params)
    return Response(content, status=status, mimetype=ctype)

@app.route('/set-sync')
def proxy_set_sync():
    params = request.args.to_dict()
    content, status, ctype = proxy_to_pi('/set-sync', params=params)
    return Response(content, status=status, mimetype=ctype)

@app.route('/cmd', methods=['POST'])
def proxy_cmd():
    data = request.get_json(silent=True) or {}
    content, status, ctype = proxy_to_pi('/cmd', method='POST', data=data)
    return Response(content, status=status, mimetype=ctype)

# ---------- Nanoplastic routes (unchanged) ----------
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

# ==================== MAIN ====================
def start_camera_thread():
    camera_thread = threading.Thread(target=camera_worker, daemon=True)
    camera_thread.start()
    time.sleep(3)

if __name__ == '__main__':
    print("=" * 60)
    print("Unified Server: USV (proxied) + Nanoplastic Detection")
    print(f"Proxying USV requests to {RASPI_URL}")
    print(f"Detection mode: {DETECTION_MODE}")
    print("=" * 60)
    start_camera_thread()
    # Start heavy metal logger thread
    hm_thread = threading.Thread(target=heavy_metal_logger, daemon=True)
    hm_thread.start()
    print("Heavy metal logger started (every 5 min).")
    print("Server running at http://localhost:5000")
    serve(app, host='0.0.0.0', port=5000, threads=4)