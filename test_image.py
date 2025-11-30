#test_image.py
from ultralytics import YOLO
import cv2, os, glob, torch
from pathlib import Path


WEIGHTS = "bestv8m.pt" 
SOURCE  = "lie174.jpg"                  
OUT_DIR = "runs/predict"
RUN_NAME = "cow-local-images"
IMG_SIZE = 640
CONF = 0.45
IOU  = 0.50
SHOW_BOXES = True

def pick_device():
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except Exception:
        pass
    return "cpu"

def collect_inputs(src):
    p = Path(src)
    if p.is_file():
        return [str(p)]
    exts = (".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp")
    if p.is_dir():
        files = [str(f) for f in p.rglob("*") if f.suffix.lower() in exts]
        if not files:
            raise FileNotFoundError(f"No images found under: {src}")
        return files
    raise FileNotFoundError(f"Path not found: {src}")

def draw_plain(annotated, boxes, names, color=(0,255,0), thickness=2, font_scale=0.8):
    if boxes is None or len(boxes) == 0:
        return annotated
    for b in boxes:
        xyxy = b.xyxy[0].tolist()
        x1,y1,x2,y2 = map(int, xyxy)
        cls_id = int(b.cls[0].item()) if b.cls is not None else -1
        label = names.get(cls_id, str(cls_id))

        if SHOW_BOXES:
            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, thickness)
        # plain text (no bg, no conf)
        y_text = max(y1 - 8, 12)
        cv2.putText(
            annotated, label, (x1, y_text),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA
        )
    return annotated

def main():
    device = pick_device()
    print(f"Using device: {device}")

    model = YOLO(WEIGHTS)
    names = model.names if hasattr(model, "names") else {}

    inputs = collect_inputs(SOURCE)
    out_root = Path(OUT_DIR) / RUN_NAME
    out_root.mkdir(parents=True, exist_ok=True)

    for path in inputs:
        img = cv2.imread(path)
        if img is None:
            print(f"Skip unreadable: {path}")
            continue

        res = model.predict(
            source=img, imgsz=IMG_SIZE, conf=CONF, iou=IOU,
            device=device, verbose=False
        )[0]

        annotated = img.copy()
        annotated = draw_plain(annotated, res.boxes, names)

        rel_name = Path(path).name
        save_path = str(out_root / rel_name)
        cv2.imwrite(save_path, annotated)
        print(f"Saved: {save_path}")

    print(f"Done. Outputs in: {out_root}")

if __name__ == "__main__":
    main()