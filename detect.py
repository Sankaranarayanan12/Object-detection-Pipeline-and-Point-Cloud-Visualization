import sys
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

SEQUENCE = sys.argv[1] if len(sys.argv) > 1 else "MVI_40751"
MODEL_WEIGHTS = "yolov8n.pt"
CONF = 0.25          
OUTPUT_FPS = 25      
SHOW = False         

# the ROI points are got from executing the pick_roi.py
ROIS = {
    "MVI_40751":  [(18, 204), (170, 451), (534, 379)],
    "MVI_41073": [(235, 159), (813, 100), (838, 483), (262, 496)],
    "MVI_63525": [(114, 99), (133, 365), (780, 306), (666, 65)],
}


VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


frame_paths = sorted(Path(SEQUENCE).glob("*.jpg"))
first = cv2.imread(str(frame_paths[0]))
h, w = first.shape[:2]

roi_points = ROIS.get(SEQUENCE, [])
if roi_points:
    roi = np.array(roi_points, dtype=np.int32)
else:
    print("No ROI set for this sequence.")
    roi = np.array([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)], dtype=np.int32)
roi_contour = roi.reshape(-1, 1, 2)


def in_roi(x1, y1, x2, y2):
    
    point = (float((x1 + x2) / 2), float(y2))
    return cv2.pointPolygonTest(roi_contour, point, False) >= 0


Path("outputs").mkdir(exist_ok=True)
out_path = f"outputs/{SEQUENCE}_demo.mp4"
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), OUTPUT_FPS, (w, h))

model = YOLO(MODEL_WEIGHTS)
results = model.predict(
    source=[str(p) for p in frame_paths],
    stream=True,                       
    conf=CONF,
    classes=list(VEHICLE_CLASSES),    
    verbose=False,
)

pre_ms, inf_ms, post_ms, total_ms = [], [], [], []
n_frames = len(frame_paths)
t_prev = time.perf_counter()

for i, result in enumerate(results):
    frame = result.orig_img.copy()

   
    overlay = frame.copy()
    cv2.fillPoly(overlay, [roi], (0, 255, 255))
    frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
    cv2.polylines(frame, [roi], True, (0, 255, 255), 2)

    count_in_roi = 0
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        if in_roi(x1, y1, x2, y2):
            count_in_roi += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = f"{VEHICLE_CLASSES[cls_id]} {conf:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 1)

    cv2.putText(frame, f"In ROI: {count_in_roi}   Frame {i + 1}/{n_frames}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    writer.write(frame)

    now = time.perf_counter()
    if i > 0:  
        pre_ms.append(result.speed["preprocess"])
        inf_ms.append(result.speed["inference"])
        post_ms.append(result.speed["postprocess"])
        total_ms.append((now - t_prev) * 1000)
    t_prev = now

writer.release()
cv2.destroyAllWindows()

if total_ms:
    print(f"\nSaved: {out_path}")
    print(f"Frames processed: {len(total_ms) + 1}")
    print(f"Mean preprocess : {np.mean(pre_ms):.1f} ms")
    print(f"Mean inference  : {np.mean(inf_ms):.1f} ms")
    print(f"Mean postprocess: {np.mean(post_ms):.1f} ms")
    print(f"Mean end-to-end : {np.mean(total_ms):.1f} ms/frame ")
    print(f"End-to-end FPS  : {1000 / np.mean(total_ms):.1f}")
