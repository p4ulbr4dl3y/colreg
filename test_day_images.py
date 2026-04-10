import os
import cv2
import glob
from colreg_vision.pipeline import VideoAnalyticsPipeline, draw_results


def main():
    # Initialize pipeline
    # Note: BinaryClassifier is heavy and loaded lazily
    pipeline = VideoAnalyticsPipeline()

    # Create output directory
    os.makedirs("test_results/day", exist_ok=True)

    # Get all images from test_images/day/
    # glob.glob('test_images/day/**/*', recursive=True) might be better if there are subdirs
    image_paths = glob.glob("test_images/day/*.*")

    print(f"Found {len(image_paths)} images to test.")

    for image_path in sorted(image_paths):
        filename = os.path.basename(image_path)
        print(f"\nProcessing {filename}...")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Failed to load {image_path}")
            continue

        # Process image
        try:
            result = pipeline.process(image, is_night=False)

            # Print results
            print(f"  Detected {result.boat_count} boats.")
            for boat in result.boats:
                status_str = ""
                if boat.day_shapes_status and boat.day_shapes_status.is_known_signal:
                    status_str = f" (Signal: {boat.day_shapes_status.vessel_type})"

                # final_vessel_type is what we care about for COLREG
                print(
                    f"  Boat {boat.boat_id}: {boat.final_vessel_type}{status_str} "
                    f"[Classification: {boat.vessel_type} {boat.vessel_type_confidence:.1f}%]"
                )

            # Draw results
            vis = draw_results(image, result)

            # Save visualization
            output_path = os.path.join("test_results/day", filename)
            cv2.imwrite(output_path, vis)
            print(f"  Saved result to {output_path}")
        except Exception as e:
            print(f"  Error processing {filename}: {e}")


if __name__ == "__main__":
    main()
