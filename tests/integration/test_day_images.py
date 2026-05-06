import os
import cv2
import glob
from colreg_vision.pipeline import VideoAnalyticsPipeline, draw_results

def main():
    pipeline = VideoAnalyticsPipeline()
    os.makedirs('test_results/day', exist_ok=True)
    image_paths = glob.glob('test_images/day/*.*')
    print(f'Found {len(image_paths)} images to test.')
    for image_path in sorted(image_paths):
        filename = os.path.basename(image_path)
        print(f'\nProcessing {filename}...')
        image = cv2.imread(image_path)
        if image is None:
            print(f'  Failed to load {image_path}')
            continue
        try:
            result = pipeline.process(image, is_night=False)
            print(f'  Detected {result.boat_count} boats.')
            for boat in result.boats:
                status_str = ''
                if boat.day_shapes_status and boat.day_shapes_status.is_known_signal:
                    status_str = f' (Signal: {boat.day_shapes_status.vessel_type})'
                print(f'  Boat {boat.boat_id}: {boat.final_vessel_type}{status_str} [Classification: {boat.vessel_type} {boat.vessel_type_confidence:.1f}%]')
            vis = draw_results(image, result)
            output_path = os.path.join('test_results/day', filename)
            cv2.imwrite(output_path, vis)
            print(f'  Saved result to {output_path}')
        except Exception as e:
            print(f'  Error processing {filename}: {e}')
if __name__ == '__main__':
    main()