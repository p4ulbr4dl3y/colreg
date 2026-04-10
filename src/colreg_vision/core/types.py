"""
Общие типы и константы для конвейера видеоаналитики.
"""


class VesselType:
    """Типы судов согласно МППСС-72."""

    MECHANICAL = "MECH"
    SAIL = "SAIL"
    FISHING = "FISH"
    NUC = "NUC"
    RAM = "RAM"
    CBD = "CBD"


# Цветовая схема для различных типов судов (BGR)
CLASS_COLORS = {
    VesselType.NUC: (0, 0, 255),  # Красный
    VesselType.RAM: (255, 0, 255),  # Маджента
    VesselType.CBD: (0, 165, 255),  # Оранжевый
    VesselType.FISHING: (0, 255, 255),  # Жёлтый
    VesselType.SAIL: (0, 255, 0),  # Зелёный
    VesselType.MECHANICAL: (255, 200, 0),  # Голубой
    "MECH": (255, 200, 0),  # Сокращение для механического
    "SAIL": (0, 255, 0),  # Сокращение для парусного
}
