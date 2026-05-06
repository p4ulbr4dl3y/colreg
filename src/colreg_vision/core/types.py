class VesselType:
    MECHANICAL = 'MECH'
    SAIL = 'SAIL'
    FISHING = 'FISH'
    NUC = 'NUC'
    RAM = 'RAM'
    CBD = 'CBD'
CLASS_COLORS = {VesselType.NUC: (0, 0, 255), VesselType.RAM: (255, 0, 255), VesselType.CBD: (0, 165, 255), VesselType.FISHING: (0, 255, 255), VesselType.SAIL: (0, 255, 0), VesselType.MECHANICAL: (255, 200, 0), 'MECH': (255, 200, 0), 'SAIL': (0, 255, 0)}