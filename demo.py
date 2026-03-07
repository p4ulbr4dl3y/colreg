"""
Demo script for video analytics pipeline.

Demonstrates vessel detection and classification on a single image.
"""

import argparse

import cv2

from pipeline import VideoAnalyticsPipeline, draw_results


def main():
    """Run demo on image.png."""
    parser = argparse.ArgumentParser(description="Video Analytics Pipeline Demo")
    parser.add_argument(
        "--image",
        type=str,
        default="image.png",
        help="Path to input image (default: image.png)",
    )
    parser.add_argument(
        "--night",
        action="store_true",
        help="Use night mode (IR + visible images)",
    )
    parser.add_argument(
        "--ir-image",
        type=str,
        help="Path to IR image for night mode",
    )
    parser.add_argument(
        "--visible-image",
        type=str,
        help="Path to visible image for night mode",
    )
    parser.add_argument(
        "--bbox-scale",
        type=float,
        nargs=4,
        default=[1.0, 1.5, 1.0, 1.0],
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="Bbox expansion factors (left, top, right, bottom). Default: 1.0 1.5 1.0 1.0",
    )
    args = parser.parse_args()

    # Initialize pipeline
    pipeline = VideoAnalyticsPipeline()

    if args.night:
        # Night mode: requires both IR and visible images
        ir_path = args.ir_image or args.image
        vis_path = args.visible_image or args.image

        print(f"Night mode:")
        print(f"  IR image: {ir_path}")
        print(f"  Visible image: {vis_path}")

        result = pipeline.process_night(
            ir_path, vis_path, bbox_scale=tuple(args.bbox_scale)
        )
    else:
        # Day mode: single image
        image_path = args.image
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Failed to load {image_path}")
            return

        print(f"Loaded image: {image_path} ({image.shape[1]}x{image.shape[0]})")
        print("\nProcessing (day mode)...")

        result = pipeline.process(image, is_night=False)

    # Print results
    print(f"\nResults:")
    print(f"  Boats detected: {result.boat_count}")

    for boat in result.boats:
        print(f"\n  Boat #{boat.boat_id}:")
        print(f"    Bbox: {boat.bbox}")
        print(
            f"    Binary classifier: {boat.vessel_type} ({boat.vessel_type_confidence:.1f}%)"
        )

        if boat.day_shapes_status:
            print(f"    Day shapes: {boat.day_shapes_status.vessel_type}")
        if boat.lights_status:
            print(f"    Lights: {boat.lights_status.vessel_type}")
        print(f"    Final type: {boat.final_vessel_type}")

    # Draw results
    vis = draw_results(result.image, result)

    # Save output
    output_path = "output.png"
    cv2.imwrite(output_path, vis)
    print(f"\nVisualization saved to: {output_path}")


if __name__ == "__main__":
    main()
