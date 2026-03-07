#!/usr/bin/env python3
"""
Example usage of the universal video analytics pipeline.

Demonstrates both individual module usage and full pipeline processing.
"""

from pathlib import Path

import cv2

from pipeline import (
    BinaryClassifier,
    Config,
    VideoAnalyticsPipeline,
    classify_day_shapes,
    classify_lights,
    detect_and_crop_boats,
    detect_infrared_objects,
)


def example_individual_modules():
    """Example: Using individual modules independently."""
    print("=" * 60)
    print("EXAMPLE: Individual Module Usage")
    print("=" * 60)

    # Load an image
    image_path = "test_image.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print(f"⚠️  Image not found: {image_path}")
        print("   Replace with your image path to run this example")
        return

    # 1. Boat detection (standalone)
    print("\n1. Boat Detection:")
    print("-" * 40)
    boat_detections = detect_and_crop_boats(image)
    print(f"   Detected {len(boat_detections)} boats")
    for i, det in enumerate(boat_detections):
        print(f"   Boat {i}: conf={det.confidence:.2f}, size={det.width}x{det.height}")

    # 2. Binary classifier (standalone)
    if boat_detections:
        print("\n2. Binary Classification:")
        print("-" * 40)
        classifier = BinaryClassifier()
        for i, det in enumerate(boat_detections[:3]):  # First 3 boats
            result = classifier.classify(det.crop)
            print(f"   Boat {i}: {result.predicted_class} ({result.confidence:.1f}%)")

    # 3. Day shapes classification (standalone)
    print("\n3. Day Shapes Classification:")
    print("-" * 40)
    day_statuses = classify_day_shapes(image)
    if day_statuses:
        for status in day_statuses:
            print(f"   Status: {status.status}, known: {status.is_known_signal}")
    else:
        print("   No day shapes detected")

    # 4. Lights classification (standalone)
    print("\n4. Lights Classification:")
    print("-" * 40)
    lights_statuses = classify_lights(image)
    if lights_statuses:
        for status in lights_statuses:
            print(f"   Status: {status.status}, sequence: {status.sequence}")
    else:
        print("   No lights detected")

    # 5. Infrared detection (standalone)
    print("\n5. Infrared Detection:")
    print("-" * 40)
    ir_detections = detect_infrared_objects(image)
    if ir_detections:
        for det in ir_detections:
            print(f"   {det.class_name}: conf={det.confidence:.2f}")
    else:
        print("   No infrared objects detected")


def example_full_pipeline():
    """Example: Using the full pipeline."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Full Pipeline Usage")
    print("=" * 60)

    # Initialize pipeline
    pipeline = VideoAnalyticsPipeline()

    # Load image
    image_path = "image.png"
    image = cv2.imread(image_path)

    if image is None:
        print(f"⚠️  Image not found: {image_path}")
        print("   Replace with your image path to run this example")
        return

    # Process in day mode
    print("\n🌞 Day Mode Processing:")
    print("-" * 40)
    result = pipeline.process(image, is_night=False)

    print(f"   Total boats: {result.boat_count}")
    print(f"   Sailboats: {result.sailboat_count}")

    for boat in result.boats:
        print(f"\n   Boat #{boat.boat_id}:")
        print(f"      Detection conf: {boat.detection_confidence:.2f}")
        print(
            f"      Vessel type: {boat.vessel_type} ({boat.vessel_type_confidence:.1f}%)"
        )
        print(f"      Final vessel type: {boat.final_vessel_type}")
        if boat.day_shapes_status:
            print(f"      Day shapes: {boat.day_shapes_status.vessel_type}")

    print(f"\n   Day shapes on full image:")
    for status in result.day_shapes_statuses:
        print(f"      - {status.vessel_type}")

    # Process in night mode
    print("\n🌙 Night Mode Processing:")
    print("-" * 40)
    result = pipeline.process(image, is_night=True)

    print(f"   Total boats: {result.boat_count}")
    print(f"   Sailboats: {result.sailboat_count}")

    for boat in result.boats:
        print(f"\n   Boat #{boat.boat_id}:")
        print(f"      Detection conf: {boat.detection_confidence:.2f}")
        print(
            f"      Vessel type: {boat.vessel_type} ({boat.vessel_type_confidence:.1f}%)"
        )
        print(f"      Final vessel type: {boat.final_vessel_type}")
        if boat.lights_status:
            print(f"      Lights: {boat.lights_status.vessel_type}")

    print(f"\n   Infrared detections: {len(result.infrared_detections)}")
    print(f"   Lights on full image: {len(result.lights_statuses)}")
    for status in result.lights_statuses:
        print(f"      - {status.vessel_type} (sequence: {status.sequence})")


def example_custom_configuration():
    """Example: Using custom configuration."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Custom Configuration")
    print("=" * 60)

    # Create custom config
    config = Config()
    config.boat_detector.confidence_threshold = 0.7  # Higher threshold
    config.lights.confidence_threshold = 0.6

    # Use custom config in pipeline
    pipeline = VideoAnalyticsPipeline(config=config)

    # Or use custom config with individual functions
    image_path = "image.png"

    boat_detections = detect_and_crop_boats(
        image=image_path, config=config, confidence_threshold=0.8  # Override config
    )

    print(f"   Using custom confidence threshold")
    print(f"   Detected {len(boat_detections)} boats")


def example_video_processing():
    """Example: Processing video stream."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Video Stream Processing")
    print("=" * 60)

    # Initialize pipeline
    pipeline = VideoAnalyticsPipeline()

    # Open video file or camera
    video_source = 0  # 0 for webcam, or "video.mp4" for file
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"⚠️  Cannot open video source: {video_source}")
        return

    frame_count = 0
    max_frames = 10  # Process only 10 frames for demo

    print(f"   Processing video... (max {max_frames} frames)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        result = pipeline.process(frame, is_night=False)

        # Print results
        print(
            f"\n   Frame {frame_count}: {result.boat_count} boats, {result.sailboat_count} sailboats"
        )

        # Draw results (optional)
        # You can add visualization code here

        frame_count += 1
        if frame_count >= max_frames:
            break

    cap.release()
    print(f"   Processed {frame_count} frames")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("VIDEO ANALYTICS PIPELINE - USAGE EXAMPLES")
    print("=" * 60)

    # Check if test image exists
    test_image = Path("image.png")
    if not test_image.exists():
        print(f"\n⚠️  Note: Create 'test_image.jpg' in current directory")
        print("   to run full examples, or modify the paths in this script.\n")

    # Run examples
    example_individual_modules()
    example_full_pipeline()
    example_custom_configuration()

    # Uncomment to run video processing
    # example_video_processing()

    print("\n" + "=" * 60)
    print("✅ Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
