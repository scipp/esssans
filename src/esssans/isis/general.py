# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Providers for the ISIS instruments.
"""
from typing import NewType, Optional

import scipp as sc

from ..common import mask_range
from ..types import (
    CleanMonitor,
    DetectorPixelShape,
    LabFrameTransform,
    MonitorType,
    NeXusMonitorName,
    RawData,
    RawMonitor,
    RunNumber,
    RunTitle,
    RunType,
    SampleRun,
    ScatteringRunType,
    UncertaintyBroadcastMode,
    WavelengthBins,
    WavelengthMonitor,
)
from ..uncertainty import broadcast_with_upper_bound_variances
from .data import LoadedFileContents

NonBackgroundWavelengthRange = NewType('NonBackgroundWavelengthRange', sc.Variable)
"""Range of wavelengths that are not considered background in the monitor"""


def get_detector_data(
    dg: LoadedFileContents[RunType],
) -> RawData[RunType]:
    return RawData[RunType](dg['data'])


def get_monitor_data(
    dg: LoadedFileContents[RunType], nexus_name: NeXusMonitorName[MonitorType]
) -> RawMonitor[RunType, MonitorType]:
    # See https://github.com/scipp/sciline/issues/52 why copy needed
    mon = dg['monitors'][nexus_name]['data'].copy()
    return RawMonitor[RunType, MonitorType](mon)


def run_number(dg: LoadedFileContents[SampleRun]) -> RunNumber:
    """Get the run number from the raw sample data."""
    return RunNumber(int(dg['run_number']))


def run_title(dg: LoadedFileContents[SampleRun]) -> RunTitle:
    """Get the run title from the raw sample data."""
    return RunTitle(dg['run_title'].value)


def helium3_tube_detector_pixel_shape() -> DetectorPixelShape[ScatteringRunType]:
    # Pixel radius and length
    # found here:
    # https://github.com/mantidproject/mantid/blob/main/instrument/SANS2D_Definition_Tubes.xml
    R = 0.00405
    L = 0.002033984375
    pixel_shape = sc.DataGroup(
        {
            'vertices': sc.vectors(
                dims=['vertex'],
                values=[
                    # Coordinates in pixel-local coordinate system
                    # Bottom face center
                    [0, 0, 0],
                    # Bottom face edge
                    [R, 0, 0],
                    # Top face center
                    [0, L, 0],
                ],
                unit='m',
            ),
            'nexus_class': 'NXcylindrical_geometry',
        }
    )
    return pixel_shape


def lab_frame_transform() -> LabFrameTransform[ScatteringRunType]:
    # Rotate +y to -x
    return sc.spatial.rotation(value=[0, 0, 1 / 2**0.5, 1 / 2**0.5])


def preprocess_monitor_data(
    monitor: WavelengthMonitor[RunType, MonitorType],
    wavelength_bins: WavelengthBins,
    non_background_range: Optional[NonBackgroundWavelengthRange],
    uncertainties: UncertaintyBroadcastMode,
) -> CleanMonitor[RunType, MonitorType]:
    """
    Prepare monitor data for computing the transmission fraction.
    The input data are first converted to wavelength (if needed).
    If a ``non_background_range`` is provided, it defines the region where data is
    considered not to be background, and regions outside are background. A mean
    background level will be computed from the background and will be subtracted from
    the non-background counts.
    Finally, if wavelength bins are provided, the data is rebinned to match the
    requested binning.

    Parameters
    ----------
    monitor:
        The monitor to be pre-processed.
    wavelength_bins:
        The binning in wavelength to use for the rebinning.
    non_background_range:
        The range of wavelengths that defines the data which does not constitute
        background. Everything outside this range is treated as background counts.
    uncertainties:
        The mode for broadcasting uncertainties. See
        :py:class:`UncertaintyBroadcastMode` for details.

    Returns
    -------
    :
        The input monitors converted to wavelength, cleaned of background counts, and
        rebinned to the requested wavelength binning.
    """
    background = None
    if non_background_range is not None:
        mask = sc.DataArray(
            data=sc.array(dims=[non_background_range.dim], values=[True]),
            coords={non_background_range.dim: non_background_range},
        )
        background = mask_range(monitor, mask=mask).mean()

    if monitor.bins is not None:
        monitor = monitor.hist(wavelength=wavelength_bins)
    else:
        monitor = monitor.rebin(wavelength=wavelength_bins)

    if background is not None:
        if uncertainties == UncertaintyBroadcastMode.drop:
            monitor -= sc.values(background)
        elif uncertainties == UncertaintyBroadcastMode.upper_bound:
            monitor -= broadcast_with_upper_bound_variances(
                background, sizes=monitor.sizes
            )
        else:
            monitor -= background
    return CleanMonitor(monitor)


providers = (
    get_detector_data,
    get_monitor_data,
    helium3_tube_detector_pixel_shape,
    lab_frame_transform,
    preprocess_monitor_data,
    run_number,
    run_title,
)
