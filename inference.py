"""
Vehicle Detection - Advanced Custom Inference Script
=====================================================
Run YOLOv8 vehicle detection on various media sources with a custom HUD 
and styled OpenCV bounding boxes.

Usage:
    python inference.py --source image.jpg --weights weights/best.pt
    python inference.py --source video.mp4 --weights weights/best.pt --conf 0.3
    python inference.py --source 0 --weights weights/best.pt --show

Author: Vishal M H
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


# Define professional colors for different vehicle classes (BGR format)
CLASS_COLORS = {
    'car': (255, 0, 0),          # Blue
    'truck': (0, 128, 255),      # Orange
    'bus': (0, 0, 255),          # Red
    'motorcycle': (0, 255, 255),  # Yellow
    'bicycle': (255, 255, 0),    # Cyan
    'auto-rickshaw': (0, 200, 100), # Green
    'van': (255, 0, 255),        # Magenta
    'ambulance': (0, 0, 150),    # Dark Red
    'taxi': (100, 255, 255),     # Light Yellow
    'scooter': (150, 150, 150),   # Gray
}
DEFAULT_COLOR = (0, 255, 0)      # Default Green for unknown classes


def parse_args():
    """Parse command-line arguments for custom inference configuration."""
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 Vehicle Detection with Custom OpenCV HUD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input / Output
    parser.add_argument("--source", type=str, required=True,
                        help="Image, video, folder path, or camera ID (0)")
    parser.add_argument("--weights", type=str, default="weights/best.pt",
                        help="Path to trained model weights")
    parser.add_argument("--save-dir", type=str, default="results/",
                        help="Directory to save annotated results")

    # Detection Parameters
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for detections")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for Non-Maximum Suppression")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size")

    # Device
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device (0 or cpu)")

    # Display Options
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save annotated results")
    parser.add_argument("--show", action="store_true", default=False,
                        help="Display results in a live window")

    return parser.parse_args()


def draw_custom_box(img, box, label, color):
    """Draw a styled semi-transparent bounding box and label tag."""
    x1, y1, x2, y2 = map(int, box)
    
    # 1. Draw glowing transparent overlay on box background
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)
    
    # 2. Draw outer border lines
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
    
    # 3. Draw a clean tag header for the class label
    (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    
    # Avoid drawing label tag outside frame borders
    tag_y1 = max(y1 - h - 10, 0)
    tag_y2 = max(y1, h + 10)
    
    tag_overlay = img.copy()
    cv2.rectangle(tag_overlay, (x1, tag_y1), (x1 + w + 10, tag_y2), color, -1)
    cv2.addWeighted(tag_overlay, 0.85, img, 0.15, 0, img)
    
    cv2.putText(img, label, (x1 + 5, tag_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(img, counts, fps=None):
    """Draw a transparent heads-up display showing model analytics."""
    h, w, _ = img.shape
    panel_w = 230
    panel_h = 45 + len(counts) * 20
    
    # Draw transparent panel background
    overlay = img.copy()
    cv2.rectangle(overlay, (15, 15), (15 + panel_w, 15 + panel_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    
    # Draw double border panel frame
    cv2.rectangle(img, (15, 15), (15 + panel_w, 15 + panel_h), (120, 120, 120), 1, lineType=cv2.LINE_AA)
    
    # Text Titles
    cv2.putText(img, "VEHICLE DETECTION HUD", (25, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(img, (25, 38), (15 + panel_w - 10, 38), (80, 80, 80), 1, cv2.LINE_AA)
    
    y = 56
    for cls_name, count in counts.items():
        text = f"{cls_name.upper()}: {count}"
        color = CLASS_COLORS.get(cls_name.lower(), DEFAULT_COLOR)
        # Draw class color indicator dot
        cv2.circle(img, (32, y - 4), 4, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(img, text, (45, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20
        
    if fps is not None:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(img, fps_text, (15 + panel_w - 65, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)


def process_image(model, img_path, save_dir, save=True, show=False, conf_thresh=0.25, iou_thresh=0.45, imgsz=640, device="0"):
    """Run prediction on a single image and apply custom styling."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Could not read image at {img_path}")
        return
        
    results = model.predict(img, conf=conf_thresh, iou=iou_thresh, imgsz=imgsz, device=device, verbose=False)[0]
    
    # Aggregate counts
    counts = {}
    
    # Parse results
    if results.boxes is not None:
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            
            # Update counts
            counts[cls_name] = counts.get(cls_name, 0) + 1
            
            # Styling
            color = CLASS_COLORS.get(cls_name.lower(), DEFAULT_COLOR)
            label = f"{cls_name} {conf:.2f}"
            draw_custom_box(img, xyxy, label, color)
            
    # Draw Dashboard HUD
    if counts:
        draw_hud(img, counts)
        
    if save:
        out_path = Path(save_dir) / Path(img_path).name
        cv2.imwrite(str(out_path), img)
        print(f"Saved: {out_path}")
        
    if show:
        cv2.imshow("Custom Vehicle Inference", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(model, source, save_dir, save=True, show=False, conf_thresh=0.25, iou_thresh=0.45, imgsz=640, device="0"):
    """Run real-time frame-by-frame prediction on video or webcam stream."""
    is_webcam = source.isdigit()
    cap = cv2.VideoCapture(int(source) if is_webcam else source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source '{source}'")
        return
        
    # Get properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS)
    if fps_src <= 0 or np.isnan(fps_src):
        fps_src = 30.0
        
    writer = None
    if save:
        out_name = "webcam_out.mp4" if is_webcam else Path(source).name
        out_path = Path(save_dir) / out_name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_src, (w, h))
        print(f"Writing output to: {out_path}")
        
    prev_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Predict on frame
        results = model.predict(frame, conf=conf_thresh, iou=iou_thresh, imgsz=imgsz, device=device, verbose=False)[0]
        
        # Calculate live FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        
        counts = {}
        if results.boxes is not None:
            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                
                counts[cls_name] = counts.get(cls_name, 0) + 1
                color = CLASS_COLORS.get(cls_name.lower(), DEFAULT_COLOR)
                label = f"{cls_name} {conf:.2f}"
                draw_custom_box(frame, xyxy, label, color)
                
        # Draw HUD dashboard onto frame
        draw_hud(frame, counts, fps)
        
        if writer is not None:
            writer.write(frame)
            
        if show:
            cv2.imshow("Custom Vehicle Inference", frame)
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def main():
    """Main custom inference pipeline."""
    args = parse_args()

    # Validate weights file exists
    if not Path(args.weights).exists():
        print(f"Error: Model weights '{args.weights}' not found.")
        sys.exit(1)

    # Initialize model
    model = YOLO(args.weights)

    # Create output directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    source_path = Path(args.source)
    
    # Determine source type and route
    if args.source.isdigit():
        print("Starting live camera inference feed...")
        process_video(model, args.source, args.save_dir, args.save, args.show, args.conf, args.iou, args.imgsz, args.device)
    elif source_path.is_file() and source_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        print(f"Starting single image inference for {source_path.name}...")
        process_image(model, source_path, args.save_dir, args.save, args.show, args.conf, args.iou, args.imgsz, args.device)
    elif source_path.is_file() and source_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
        print(f"Starting video stream inference for {source_path.name}...")
        process_video(model, args.source, args.save_dir, args.save, args.show, args.conf, args.iou, args.imgsz, args.device)
    elif source_path.is_dir():
        print(f"Starting batch folder inference for directory '{source_path}'...")
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        img_files = [f for f in source_path.iterdir() if f.suffix.lower() in image_extensions]
        
        if not img_files:
            print(f"No matching image files found in {source_path}")
            sys.exit(0)
            
        for img_file in img_files:
            process_image(model, img_file, args.save_dir, args.save, False, args.conf, args.iou, args.imgsz, args.device)
    else:
        print(f"Error: Unrecognized source type: '{args.source}'")
        sys.exit(1)


if __name__ == "__main__":
    main()
