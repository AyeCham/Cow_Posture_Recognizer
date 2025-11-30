# test_video.py
from ultralytics import YOLO
import cv2, os, time, torch

WEIGHTS = "bestv8m.pt"      
SOURCE  = "test1.mp4"   
OUT_DIR = "runs/predict"
RUN_NAME = "cow-local-video"

def pick_device():
    
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except Exception:
        pass
    return "cpu"

def main():
    device = pick_device()
    print(f"Using device: {device}")

    model = YOLO(WEIGHTS)

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {SOURCE}")

    os.makedirs(f"{OUT_DIR}/{RUN_NAME}", exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 1:
        fps = 30.0

    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    out_path = f"{OUT_DIR}/{RUN_NAME}/output.mp4"
    for fourcc_tag in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_tag)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if writer.isOpened():
            break
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter. Try installing FFmpeg or use a different codec/container.")

 
    IMG = 640     
    CONF = 0.6
    IOU = 0.5
    VID_STRIDE = 2

    t0, frames = time.time(), 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        if VID_STRIDE > 1 and (frames % VID_STRIDE):
            continue

        results = model.predict(
            source=frame, imgsz=IMG, conf=CONF, iou=IOU,
            device=device, verbose=False
        )

        r = results[0]
        annotated = frame.copy()

        # Class name map
        names = model.names if hasattr(model, "names") else {i: str(i) for i in range(1000)}

        # Draw boxes + plain text (no background, no confidence)
        for b in (r.boxes or []):
            xyxy = b.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            cls_id = int(b.cls[0].item()) if b.cls is not None else -1
            label = names.get(cls_id, str(cls_id))

            # box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # text (no background rectangle)
            y_text = max(y1 - 8, 12)
            cv2.putText(
                annotated, label, (x1, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
            )



        if frames % 10 == 0:
            fps_live = frames / max(time.time() - t0, 1e-5)
            cv2.putText(annotated, f"FPS: {fps_live:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        writer.write(annotated)
        cv2.imshow("Predictions (q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
