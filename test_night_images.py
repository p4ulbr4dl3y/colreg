import os
import cv2
from pipeline import VideoAnalyticsPipeline, draw_results


def main():
    # Initialize pipeline
    pipeline = VideoAnalyticsPipeline()

    # Create output directory
    os.makedirs("test_results/night", exist_ok=True)

    # Path to night test images
    night_dir = "test_images/night"

    # Categories of night images
    categories = ["cbd", "fishing", "nuc", "power-driven", "ram", "sailboat"]

    print("Starting Night Mode Tests...")

    for cat in categories:
        cat_dir = os.path.join(night_dir, cat)
        ir_path = os.path.join(cat_dir, "ir.png")

        # Handle the 'nomal.png' typo in ram directory
        visible_path = os.path.join(cat_dir, "normal.png")
        if not os.path.exists(visible_path):
            visible_path = os.path.join(cat_dir, "nomal.png")

        if not os.path.exists(ir_path) or not os.path.exists(visible_path):
            print(f"\nMissing files in {cat_dir}, skipping.")
            continue

        print(f"\nProcessing category: {cat}")
        print(f"  IR: {ir_path}")
        print(f"  Visible: {visible_path}")

        # Load images
        ir_img = cv2.imread(ir_path)
        vis_img = cv2.imread(visible_path)

        if ir_img is None or vis_img is None:
            print(f"  Failed to load images for {cat}")
            continue

        # Process night (uses IR for detection, Visible for lights)
        try:
            # bbox_scale is already (1.0, 5.0, 1.0, 1.0) by default in pipeline.py
            result = pipeline.process_night(ir_img, vis_img)

            # Print results
            print(f"  Detected {result.boat_count} boats.")
            for boat in result.boats:
                status_str = ""
                if boat.lights_status and boat.lights_status.is_known_signal:
                    status_str = f" (Lights: {boat.lights_status.vessel_type})"

                print(
                    f"  Boat {boat.boat_id}: {boat.final_vessel_type}{status_str} "
                    f"[Classification: {boat.vessel_type} {boat.vessel_type_confidence:.1f}%]"
                )

            # Draw results on IR image (as per process_night logic)
            vis_result = draw_results(ir_img, result)

            # Save visualization
            output_path = os.path.join("test_results/night", f"{cat}_result.png")
            cv2.imwrite(output_path, vis_result)
            print(f"  Saved result to {output_path}")

        except Exception as e:
            print(f"  Error processing {cat}: {e}")


if __name__ == "__main__":
    main()
