import os

import cv2
import pytest

from colreg_vision.pipeline import VideoAnalyticsPipeline


@pytest.fixture(scope="session")
def pipeline():
    return VideoAnalyticsPipeline()


@pytest.mark.parametrize(
    "image_name, expected_type",
    [
        ("cbd.png", "CBD"),
        ("fishing.png", "FISH"),
        ("nuc.png", "NUC"),
        ("power-driven.webp", "MECH"),
        ("ram.png", "RAM"),
        ("sailing.jpg", "SAIL"),
    ],
)
def test_day_mode_classifications(pipeline, image_name, expected_type):
    image_path = os.path.join("test_images/day", image_name)
    assert os.path.exists(image_path), f"Тестовое изображение {image_path} не найдено"
    image = cv2.imread(image_path)
    result = pipeline.process(image, is_night=False)
    assert result.boat_count > 0, f"На изображении {image_name} не найдено судов"
    found_expected = any(
        (boat.final_vessel_type == expected_type for boat in result.boats)
    )
    assert found_expected, (
        f"Ожидаемый тип {expected_type} не найден на {image_name}. Найденные типы: {[b.final_vessel_type for b in result.boats]}"
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
    cat_dir = os.path.join("test_images/night", category)
    ir_path = os.path.join(cat_dir, "ir.png")
    visible_path = os.path.join(cat_dir, "normal.png")
    if not os.path.exists(visible_path):
        visible_path = os.path.join(cat_dir, "nomal.png")
    assert os.path.exists(ir_path), f"ИК-изображение {ir_path} не найдено"
    assert os.path.exists(visible_path), (
        f"Видимое изображение {visible_path} не найдено"
    )
    ir_image = cv2.imread(ir_path)
    visible_image = cv2.imread(visible_path)
    result = pipeline.process_night(ir_image, visible_image)
    assert result.boat_count > 0, f"В категории {category} не найдено судов"
    found_expected = any(
        (boat.final_vessel_type == expected_type for boat in result.boats)
    )
    assert found_expected, (
        f"Ожидаемый тип {expected_type} не найден в ночной категории {category}. Найденные типы: {[b.final_vessel_type for b in result.boats]}"
    )


def test_empty_image(pipeline):
    image_path = "test_images/day/sea.webp"
    image = cv2.imread(image_path)
    result = pipeline.process(image, is_night=False)
    assert result.boat_count == 0
    assert result.sailboat_count == 0
    assert result.mechanical_count == 0
