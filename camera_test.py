import cv2
import numpy as np

def test_camera(index=0, backend=cv2.CAP_ANY, width=None, height=None, fourcc=None):
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        print("  Failed to open")
        return None
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    # Try to enable auto-exposure (0.25 = manual, 0.75 = auto, 1 = auto??)
    # Most common: set to 3 (auto) or 1 (auto)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)   # auto
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)        # often -6 = auto for some cams
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)     # neutral brightness

    for _ in range(20):
        ret, frame = cap.read()
        if ret and frame is not None:
            mean_val = np.mean(frame)
            if mean_val > 15:   # reasonably bright
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                return (actual_w, actual_h, mean_val)
    cap.release()
    return None

backends = [
    ("Default", cv2.CAP_ANY),
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
]

resolutions = [
    (None, None, "native"),
    (640, 480, "640x480"),
    (1280, 720, "1280x720"),
]

fourccs = [
    (None, "default"),
    (cv2.VideoWriter_fourcc(*'MJPG'), "MJPG"),
    (cv2.VideoWriter_fourcc(*'YUYV'), "YUYV"),
]

print("Camera diagnostic – finding working settings")
print("=" * 50)
for idx in range(3):
    for bname, bval in backends:
        for w, h, res_name in resolutions:
            for fourcc, fname in fourccs:
                res = test_camera(idx, bval, w, h, fourcc)
                if res:
                    print(f"✓ Index {idx} | {bname:7s} | {res_name:8s} | {fname:5s} → {res[0]}x{res[1]} mean={res[2]:.1f}")