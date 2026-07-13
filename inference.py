"""
Vehicle Detection - Inference Script
=====================================
Run YOLOv8 vehicle detection on images, videos, or camera feeds.

Usage:
    python inference.py --source image.jpg --weights weights/best.pt
    python inference.py --source video.mp4 --weights weights/best.pt --conf 0.3
    python inference.py --source 0 --weights weights/best.pt --show

Author: Vishal M H
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    """Parse command-line arguments for inference configuration."""
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 Vehicle Detection Inference",
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
    parser.add_argument("--max-det", type=int, default=300,
                        help="Maximum number of detections per image")

    # Device
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device (0, cpu)")

    # Display Options
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save annotated results")
    parser.add_argument("--show", action="store_true", default=False,
                        help="Display results in a window")
    parser.add_argument("--show-labels", action="store_true", default=True,
                        help="Show class labels on detections")
    parser.add_argument("--show-conf", action="store_true", default=True,
                        help="Show confidence scores on detections")
    parser.add_argument("--line-width", type=int, default=2,
                        help="Bounding box line width")

    return parser.parse_args()


def main():
    """Main inference pipeline."""
    args = parse_args()

    # Validate weights file exists
    if not Path(args.weights).exists():
        print(f"Error: Model weights '{args.weights}' not found.")
        print("Download or train a model first. See README.md for instructions.")
        sys.exit(1)

    # Validate source
    if not args.source.isdigit() and not Path(args.source).exists():
        print(f"Error: Source '{args.source}' not found.")
        sys.exit(1)

    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Vehicle Detection - YOLOv8 Inference")
    print("=" * 60)
    print(f"  Source     : {args.source}")
    print(f"  Weights    : {args.weights}")
    print(f"  Confidence : {args.conf}")
    print(f"  IoU        : {args.iou}")
    print(f"  Image Size : {args.imgsz}")
    print(f"  Device     : {args.device}")
    print("=" * 60)

    # Load model
    model = YOLO(args.weights)

    # Run inference
    results = model.predict(
        source=int(args.source) if args.source.isdigit() else args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
        device=args.device,
        save=args.save,
        show=args.show,
        show_labels=args.show_labels,
        show_conf=args.show_conf,
        line_width=args.line_width,
        project=args.save_dir,
        name="detect",
        exist_ok=True,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("  Inference Complete!")
    print(f"  Processed {len(results)} frame(s)")
    if args.save:
        print(f"  Results saved to: {args.save_dir}/detect/")
    print("=" * 60)


if __name__ == "__main__":
    main()
