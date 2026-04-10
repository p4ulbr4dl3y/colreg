import pytest
import os
import cv2
from pipeline import VideoAnalyticsPipeline


@pytest.fixture(scope="module")
def pipeline():
    """Фикстура для однократной инициализации конвейера для всех тестов."""
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
    """Проверка того, что дневные изображения классифицируются правильно по их сигналам или бинарным классом."""
    image_path = os.path.join("test_images/day", image_name)
    assert os.path.exists(image_path), f"Тестовое изображение {image_path} не найдено"

    image = cv2.imread(image_path)
    result = pipeline.process(image, is_night=False)

    assert result.boat_count > 0, f"На изображении {image_name} не найдено судов"

    # Проверяем, совпадает ли классификация хотя бы одного судна с ожидаемой
    # На некоторых изображениях несколько судов, проверяем корректность целевого судна
    found_expected = any(
        boat.final_vessel_type == expected_type for boat in result.boats
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
    """Проверка того, что ночные изображения (ИК + видимый спектр) классифицируются правильно."""
    cat_dir = os.path.join("test_images/night", category)
    ir_path = os.path.join(cat_dir, "ir.png")

    # Обрабатываем опечатку 'nomal.png' в директории ram
    visible_path = os.path.join(cat_dir, "normal.png")
    if not os.path.exists(visible_path):
        visible_path = os.path.join(cat_dir, "nomal.png")

    assert os.path.exists(ir_path), f"ИК изображение не найдено в директории {cat_dir}"
    assert os.path.exists(visible_path), (
        f"Изображение видимого спектра не найдено в директории {cat_dir}"
    )

    ir_img = cv2.imread(ir_path)
    vis_img = cv2.imread(visible_path)

    result = pipeline.process_night(ir_img, vis_img)

    assert result.boat_count > 0, f"Суда не найдены в ночной категории {category}"

    found_expected = any(
        boat.final_vessel_type == expected_type for boat in result.boats
    )
    assert found_expected, (
        f"Ожидаемый тип {expected_type} не найден в категории {category}. Найденные типы: {[b.final_vessel_type for b in result.boats]}"
    )


def test_empty_sea_scene(pipeline):
    """Проверка того, что на пустом изображении моря не обнаруживаются суда."""
    image_path = "test_images/day/sea.webp"
    assert os.path.exists(image_path)

    image = cv2.imread(image_path)
    result = pipeline.process(image, is_night=False)

    assert result.boat_count == 0, (
        f"Ложное срабатывание: обнаружено {result.boat_count} судов на пустом морском пейзаже!"
    )
