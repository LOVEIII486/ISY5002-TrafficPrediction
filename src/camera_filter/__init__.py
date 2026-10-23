"""Camera filter: remove fixed false positives from the detector output using a per-camera,
human-calibrated road region and exclusion set.

    from src.camera_filter import CameraFilter, FilteredDetector
"""
from .core import CameraFilter, CameraRules, Exclusion, points_in_polygon  # noqa: F401


def __getattr__(name):  # FilteredDetector needs ultralytics, so import it only when used
    if name == "FilteredDetector":
        from .detector import FilteredDetector

        return FilteredDetector
    raise AttributeError(name)
