import pytest
import os
import cv2
from pipeline import VideoAnalyticsPipeline


@pytest.fixture(scope="module")
def pipeline():
    """Fixture to initialize the pipeline once for all tests."""
    return VideoAnalyticsPipeline()


@pytest.mark.parametrize(
    "image_name, expected_type",
    [
        ("cbd.png", "CBD"),
        ("fishing.png", "FISH"),
        ("nuc.png", "NUC"),
        ("power-driven.webp", "CBD"),
        ("ram.png", "RAM"),
        ("sailing.jpg", "SAIL"),
    ],
)
def test_day_mode_classifications(pipeline, image_name, expected_type):
    """Test that day images are classified correctly by their signals or binary class."""
    image_path = os.path.join("test_images/day", image_name)
    assert os.path.exists(image_path), f"Test image {image_path} not found"

    image = cv2.imread(image_path)
    result = pipeline.process(image, is_night=False)

    assert result.boat_count > 0, f"No boats detected in {image_name}"

    # Check if at least one boat in the image matches the expected classification
    # Some images have multiple boats, so we check if the primary one is correct
    found_expected = any(
        boat.final_vessel_type == expected_type for boat in result.boats
    )
    assert found_expected, (
        f"Expected {expected_type} not found in {image_name}. Types found: {[b.final_vessel_type for b in result.boats]}"
    )


@pytest.mark.parametrize(
    "category, expected_type",
    [
        ("cbd", "CBD"),
        ("fishing", "FISH"),
        ("nuc", "NUC"),
        ("power-driven", "MECH"),
        ("ram", "RAM"),
        ("sailboat", "SAIL"),
    ],
)
def test_night_mode_classifications(pipeline, category, expected_type):
    """Test that night images (IR + Visible) are classified correctly."""
    cat_dir = os.path.join("test_images/night", category)
    ir_path = os.path.join(cat_dir, "ir.png")

    # Handle the 'nomal.png' typo in ram directory
    visible_path = os.path.join(cat_dir, "normal.png")
    if not os.path.exists(visible_path):
        visible_path = os.path.join(cat_dir, "nomal.png")

    assert os.path.exists(ir_path), f"IR image not found in {cat_dir}"
    assert os.path.exists(visible_path), f"Visible image not found in {cat_dir}"

    ir_img = cv2.imread(ir_path)
    vis_img = cv2.imread(visible_path)

    result = pipeline.process_night(ir_img, vis_img)

    assert result.boat_count > 0, f"No boats detected in night category {category}"

    found_expected = any(
        boat.final_vessel_type == expected_type for boat in result.boats
    )
    assert found_expected, (
        f"Expected {expected_type} not found in {category}. Types found: {[b.final_vessel_type for b in result.boats]}"
    )


def test_empty_sea_scene(pipeline):
    """Test that no boats are detected in an empty sea scene."""
    image_path = "test_images/day/sea.webp"
    assert os.path.exists(image_path)

    image = cv2.imread(image_path)
    result = pipeline.process(image, is_night=False)

    assert result.boat_count == 0, (
        f"Detected {result.boat_count} boats in an empty sea scene!"
    )
