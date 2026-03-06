#!/usr/bin/env python3
"""Test script for the video analytics pipeline using image.png"""

import sys
from pathlib import Path

import cv2

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from binary_classifier import BinaryClassifier
from boat_detector import detect_and_crop_boats

# Import modules directly (not as package)
from config import Config
from day_shapes import classify_day_shapes
from infrared_detector import detect_infrared_objects
from lights import classify_lights
from pipeline import VideoAnalyticsPipeline


def main():
    print("=" * 60)
    print("Testing Video Analytics Pipeline with image.png")
    print("=" * 60)

    # Load image
    image_path = Path(__file__).parent / "image.png"
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Failed to load image: {image_path}")
        return

    print(f"\n✓ Image loaded: {image_path}")
    print(f"  Dimensions: {image.shape[1]}x{image.shape[0]}")

    # Test 1: Boat detection only
    print("\n" + "-" * 40)
    print("Test 1: Boat Detection")
    print("-" * 40)
    try:
        boat_detections = detect_and_crop_boats(image)
        print(f"✓ Boat detection completed")
        print(f"  Detected {len(boat_detections)} boats")
        for i, det in enumerate(boat_detections):
            print(
                f"    Boat {i}: conf={det.confidence:.2f}, size={det.width}x{det.height}"
            )
    except FileNotFoundError as e:
        print(f"⚠ Model file not found: {e}")
    except Exception as e:
        print(f"✗ Boat detection failed: {e}")

    # Test 2: Full pipeline (Day Mode)
    print("\n" + "-" * 40)
    print("Test 2: Full Pipeline (Day Mode)")
    print("-" * 40)
    try:
        pipeline = VideoAnalyticsPipeline()
        result = pipeline.process(image, is_night=False)
        print(f"✓ Pipeline processing completed")
        print(f"  Total boats: {result.boat_count}")
        print(f"  Sailboats: {result.sailboat_count}")

        for boat in result.boats:
            print(f"\n  Boat #{boat.boat_id}:")
            print(f"    Detection conf: {boat.detection_confidence:.2f}")
            print(
                f"    Is sailboat: {boat.is_sailboat} ({boat.sailboat_confidence:.1f}%)"
            )
            if boat.day_shapes_status:
                print(f"    Day shapes: {boat.day_shapes_status.status}")

        if result.day_shapes_statuses:
            print(f"\n  Day shapes on full image:")
            for status in result.day_shapes_statuses:
                print(f"    - {status.status}")

    except FileNotFoundError as e:
        print(f"⚠ Model file not found: {e}")
    except Exception as e:
        print(f"✗ Pipeline failed: {e}")

    # Test 3: Night mode
    print("\n" + "-" * 40)
    print("Test 3: Full Pipeline (Night Mode)")
    print("-" * 40)
    try:
        pipeline = VideoAnalyticsPipeline()
        result = pipeline.process(image, is_night=True)
        print(f"✓ Pipeline processing completed")
        print(f"  Total boats: {result.boat_count}")
        print(f"  Sailboats: {result.sailboat_count}")
        print(f"  Infrared detections: {len(result.infrared_detections)}")
        print(f"  Lights statuses: {len(result.lights_statuses)}")

    except FileNotFoundError as e:
        print(f"⚠ Model file not found: {e}")
    except Exception as e:
        print(f"✗ Pipeline failed: {e}")

    print("\n" + "=" * 60)
    print("Testing completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
