"""Headless Open3D surface needed by the G1 SLAM localizer.

The robot's Open3D build enables GUI support, so its stock top-level import
unconditionally loads Plotly and Dash. SLAM uses only the native CPU geometry,
I/O, utility, and registration bindings.
"""

from open3d.cpu.pybind import camera, data, geometry, io, pipelines, t, utility
from open3d.cpu import pybind

__DEVICE_API__ = "cpu"
__version__ = "0.16.0"

__all__ = [
    "camera",
    "data",
    "geometry",
    "io",
    "pipelines",
    "pybind",
    "t",
    "utility",
]
