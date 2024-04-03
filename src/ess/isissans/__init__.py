# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

# flake8: noqa E402
import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from . import components, data, general, io, sans2d
from .components import DetectorBankOffset, MonitorOffset, SampleOffset
from .io import CalibrationFilename
from .visualization import plot_flat_detector_xy

providers = components.providers + general.providers + io.providers


__all__ = [
    'CalibrationFilename',
    'DetectorBankOffset',
    'apply_component_user_offsets_to_raw_data',
    'data',
    'io',
    'MonitorOffset',
    'providers',
    'SampleOffset',
    'plot_flat_detector_xy',
    'sans2d',
]
