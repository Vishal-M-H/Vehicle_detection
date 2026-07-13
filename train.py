"""
Vehicle Detection - Local Training Script
==========================================
Train YOLOv8 model for road vehicle detection.

Usage:
    python train.py --data configs/data.yaml --epochs 100 --imgsz 640
    python train.py --data configs/data.yaml --model yolov8m.pt --epochs 150 --batch 16

Author: Vishal M H
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model for Vehicle Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model & Data
    parser.add_argument("--data", type=str, default="configs/data.yaml",
                        help="Path to data configuration YAML file")
    parser.add_argument("--model", type=str, default="yolov8m.pt",
                        help="Pretrained model to use (yolov8n/s/m/l/x.pt)")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (-1 for auto)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="Initial learning rate")
    parser.add_argument("--optimizer", type=str, default="auto",
                        choices=["SGD", "Adam", "AdamW", "auto"],
                        help="Optimizer to use")

    # Device & Performance
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device (0, 0,1, cpu)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of data loader workers")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Enable Automatic Mixed Precision")
    parser.add_argument("--cache", type=str, default="ram",
                        choices=["ram", "disk", "False"],
                        help="Cache images for faster training")

    # Output
    parser.add_argument("--project", type=str, default="runs/train",
                        help="Project directory for saving results")
    parser.add_argument("--name", type=str, default="vehicle_det",
                        help="Experiment name")
    parser.add_argument("--save-period", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")

    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()

    # Validate data config exists
    if not Path(args.data).exists():
        print(f"Error: Data config '{args.data}' not found.")
        sys.exit(1)

    print("=" * 60)
    print("  Vehicle Detection - YOLOv8 Training")
    print("=" * 60)
    print(f"  Model      : {args.model}")
    print(f"  Data       : {args.data}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Image Size : {args.imgsz}")
    print(f"  Batch Size : {args.batch}")
    print(f"  Device     : {args.device}")
    print("=" * 60)

    # Load model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        amp=args.amp,
        cache=args.cache if args.cache != "False" else False,
        project=args.project,
        name=args.name,
        save_period=args.save_period,
        seed=args.seed,
        exist_ok=True,
        verbose=True,
        plots=True,
        resume=args.resume,
    )

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print(f"  Best model saved at: {results.save_dir}/weights/best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
