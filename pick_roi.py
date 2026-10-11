import sys
from pathlib import Path
import cv2
import numpy as np

sequence = sys.argv[1] if len(sys.argv) > 1 else "MVI_40751"
frame_paths = sorted(Path(sequence).glob("*.jpg"))
frame = cv2.imread(str(frame_paths[len(frame_paths) // 2]))
points = []


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))


cv2.namedWindow("pick ROI")
cv2.setMouseCallback("pick ROI", on_click)

while True:
    view = frame.copy()
    for p in points:
        cv2.circle(view, p, 4, (0, 0, 255), -1)
    if len(points) >= 2:
        cv2.polylines(view, [np.array(points, dtype=np.int32)],
                      len(points) >= 3, (0, 255, 255), 2)
    cv2.imshow("pick ROI", view)
    key = cv2.waitKey(20) & 0xFF
    if key == ord("u") and points:
        points.pop()
    elif key in (13, ord("q")):
        break

cv2.destroyAllWindows()
print("ROI_POINTS =", points)