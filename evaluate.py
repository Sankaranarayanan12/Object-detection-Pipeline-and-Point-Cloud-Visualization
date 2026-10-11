import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


SEQUENCES = sys.argv[1:] or ["MVI_40751", "MVI_41073", "MVI_63525"]
MODELS = ["yolov8n.pt"]        
ANNO_DIR = "."                 
EVAL_CONF = 0.001              
OP_CONF = 0.25                
NMS_IOU = 0.6
IOU_MATCH = 0.5                
EVAL_CLASSES = [2, 5, 7]       
IGNORE_OVERLAP = 0.5           
N_FAILURE_FRAMES = 5


ROIS = {
    "MVI_40751":  [(18, 204), (170, 451), (534, 379)],
    "MVI_41073": [(235, 159), (813, 100), (838, 483), (262, 496)],
    "MVI_63525": [(114, 99), (133, 365), (780, 306), (666, 65)],
}



def parse_xml(xml_path):
    root = ET.parse(xml_path).getroot()

    attrs = root.find("sequence_attribute")
    weather = "?"
    if attrs is not None:
        weather = attrs.attrib.get("sence_weather") or "?"

    ignored = []
    for b in root.findall("ignored_region/box"):
        l, t = float(b.attrib["left"]), float(b.attrib["top"])
        ignored.append([l, t, l + float(b.attrib["width"]), t + float(b.attrib["height"])])

    gt = {}
    for frame in root.findall("frame"):
        boxes = []
        for target in frame.findall("target_list/target"):
            b = target.find("box")
            l, t = float(b.attrib["left"]), float(b.attrib["top"])
            boxes.append([l, t, l + float(b.attrib["width"]), t + float(b.attrib["height"])])
        gt[int(frame.attrib["num"])] = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    return gt, ignored, weather


def frame_number(path, fallback):
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else fallback


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def ignored_fraction(boxes, ignored):
    if len(boxes) == 0 or not ignored:
        return np.zeros(len(boxes))
    area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-6)
    inter = np.zeros(len(boxes))
    for ig in ignored:
        w = np.clip(np.minimum(boxes[:, 2], ig[2]) - np.maximum(boxes[:, 0], ig[0]), 0, None)
        h = np.clip(np.minimum(boxes[:, 3], ig[3]) - np.maximum(boxes[:, 1], ig[1]), 0, None)
        inter += w * h
    return np.minimum(inter / area, 1.0)


def roi_mask(boxes, contour):
    
    return np.array(
        [cv2.pointPolygonTest(contour, (float((b[0] + b[2]) / 2), float(b[3])), False) >= 0
         for b in boxes],
        dtype=bool,
    )


def match_frame(pred_boxes, gt_boxes):
   
    tp = np.zeros(len(pred_boxes), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return tp
    ious = iou_matrix(pred_boxes, gt_boxes)
    used = np.zeros(len(gt_boxes), dtype=bool)
    for i in range(len(pred_boxes)):
        cand = np.where(used, -1.0, ious[i])
        j = int(np.argmax(cand))
        if cand[j] >= IOU_MATCH:
            tp[i] = True
            used[j] = True
    return tp


def average_precision(conf, tp, n_gt):

    if n_gt == 0 or len(conf) == 0:
        return 0.0
    order = np.argsort(-conf)
    tp = tp[order].astype(float)
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1.0 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def summarize(conf_list, tp_list, n_gt):
    conf = np.concatenate(conf_list) if conf_list else np.zeros(0)
    tp = np.concatenate(tp_list) if tp_list else np.zeros(0, dtype=bool)
    ap = average_precision(conf, tp, n_gt)
    hi = conf >= OP_CONF
    tp_n = int(np.sum(tp & hi))
    fp_n = int(np.sum(hi)) - tp_n
    fn_n = n_gt - tp_n
    precision = tp_n / (tp_n + fp_n) if (tp_n + fp_n) else 0.0
    recall = tp_n / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return ap, precision, recall, f1, tp_n, fp_n, fn_n


def evaluate_sequence(seq, model, model_name):
    frame_paths = sorted(Path(seq).glob("*.jpg"))
    if not frame_paths:
        raise SystemExit(f"No .jpg frames found in: {seq}")
    xml_path = Path(ANNO_DIR) / f"{seq}.xml"
    if not xml_path.exists():
        raise SystemExit(f"Annotation file not found: {xml_path}")

    gt_all, ignored, weather = parse_xml(xml_path)
    frame_nums = [frame_number(p, i + 1) for i, p in enumerate(frame_paths)]
    missing = sum(1 for n in frame_nums if n not in gt_all)
    print(f"\n[{model_name}] {seq} ({weather}): {len(frame_paths)} frames, "
          f"{len(ignored)} ignored regions, {missing} frames without annotations")

    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]
    pts = ROIS.get(seq, [])
    roi = (np.array(pts, dtype=np.int32) if pts
           else np.array([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)], dtype=np.int32))
    roi_contour = roi.reshape(-1, 1, 2)

    results = model.predict(
        source=[str(p) for p in frame_paths],
        stream=True,
        conf=EVAL_CONF,
        iou=NMS_IOU,
        classes=EVAL_CLASSES,
        max_det=300,
        verbose=False,
    )

    scopes = {s: {"conf": [], "tp": [], "n_gt": 0} for s in ("full", "roi")}
    per_frame = []
    inf_ms = []

    for i, result in enumerate(results):
        gt = gt_all.get(frame_nums[i], np.zeros((0, 4), dtype=np.float32))

        if len(result.boxes):
            boxes = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            order = np.argsort(-conf)
            boxes, conf = boxes[order], conf[order]
            keep = ignored_fraction(boxes, ignored) <= IGNORE_OVERLAP
            boxes, conf = boxes[keep], conf[keep]
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            conf = np.zeros(0, dtype=np.float32)

        for scope in ("full", "roi"):
            if scope == "roi":
                pm, gm = roi_mask(boxes, roi_contour), roi_mask(gt, roi_contour)
                b, c, g = boxes[pm], conf[pm], gt[gm]
            else:
                b, c, g = boxes, conf, gt
            tp = match_frame(b, g)
            S = scopes[scope]
            S["conf"].append(c)
            S["tp"].append(tp)
            S["n_gt"] += len(g)

            if scope == "full":
                hi = c >= OP_CONF
                tp_f = int(np.sum(tp & hi))
                fp_f = int(np.sum(hi)) - tp_f
                fn_f = len(g) - tp_f
                per_frame.append((fp_f + fn_f, i, b[hi], g))

        if i > 0:  
            inf_ms.append(result.speed["inference"])

    
    per_frame.sort(key=lambda x: -x[0])
    for rank, (score, i, pred_b, gt_b) in enumerate(per_frame[:N_FAILURE_FRAMES]):
        img = cv2.imread(str(frame_paths[i]))
        for x1, y1, x2, y2 in gt_b.astype(int):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        for x1, y1, x2, y2 in pred_b.astype(int):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 1)
        cv2.putText(img, f"red=GT green=pred(conf>={OP_CONF}) fp+fn={score}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(f"outputs/failures/{model_name}_{seq}_{rank + 1}_frame{frame_nums[i]}.jpg", img)

    rows = []
    for scope, S in scopes.items():
        ap, p, r, f1, tp_n, fp_n, fn_n = summarize(S["conf"], S["tp"], S["n_gt"])
        rows.append({
            "model": model_name, "sequence": seq, "weather": weather, "scope": scope,
            "frames": len(frame_paths), "gt_boxes": S["n_gt"],
            "AP50": round(ap, 4), "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f1, 4), "TP": tp_n, "FP": fp_n, "FN": fn_n,
            "infer_ms": round(float(np.mean(inf_ms)), 1) if inf_ms else 0.0,
        })
    return rows


def main():
    Path("outputs/failures").mkdir(parents=True, exist_ok=True)
    rows = []
    for weights in MODELS:
        model = YOLO(weights)
        for seq in SEQUENCES:
            rows.extend(evaluate_sequence(seq, model, Path(weights).stem))

    print(f"\nPrecision/recall/F1 at conf >= {OP_CONF}; AP50 from conf >= {EVAL_CONF}; "
          f"IoU {IOU_MATCH}; class-agnostic over COCO classes {EVAL_CLASSES}")
    print(f"{'model':<9}{'sequence':<11}{'weather':<8}{'scope':<6}{'GT':>7}{'AP50':>8}"
          f"{'P':>8}{'R':>8}{'F1':>8}{'ms/img':>8}")
    for r in rows:
        print(f"{r['model']:<9}{r['sequence']:<11}{r['weather']:<8}{r['scope']:<6}{r['gt_boxes']:>7}"
              f"{r['AP50']:>8.3f}{r['precision']:>8.3f}{r['recall']:>8.3f}{r['f1']:>8.3f}"
              f"{r['infer_ms']:>8.1f}")

    with open("outputs/eval_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved outputs/eval_results.csv and outputs/failures/*.jpg")


if __name__ == "__main__":
    main()