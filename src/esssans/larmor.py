# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Loading and masking specific to the ISIS Sans2d instrument and files stored in Scipp's
HDF5 format.
"""
from typing import Optional, NewType, TypeVar

import sciline
import scipp as sc
import scippneutron as scn

from .common import gravity_vector
from .types import (
    DirectBeamNumberOfSamplingPoints,
    DirectBeamSamplingWavelengthWidth,
    DirectBeamWavelengthSamplingPoints,
    Filename,
    MaskedData,
    NeXusMonitorName,
    MonitorType,
    RawData,
    RawMonitor,
    RunType,
    SampleRun,
    SourcePosition,
    WavelengthBins,
)


DetectorStrawMask = NewType('DetectorStrawMask', sc.Variable)
"""Detector straw mask"""
DetectorBeamStopMask = NewType('DetectorBeamStopMask', sc.Variable)
"""Detector beam stop mask"""
DetectorTubeEdgeMask = NewType('DetectorTubeEdgeMask', sc.Variable)
"""Detector tube edge mask"""


class DataAsStraws(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data reshaped to have a straw dimension"""


def load_larmor_run(filename: Filename[RunType]) -> RawData[RunType]:
    # from .data import get_path

    # da = scn.load_nexus(filename=get_path(filename))
    da = scn.load_nexus(filename)
    da.coords['sample_position'] = sc.vector([0, 0, 0], unit='m')
    da.bins.constituents['data'].variances = da.bins.constituents['data'].values
    for name in ('monitor_1', 'monitor_2'):
        monitor = da.attrs[name].value
        if 'source_position' not in monitor.coords:
            monitor.coords["source_position"] = da.coords['source_position']
        monitor.values[0].variances = monitor.values[0].values
    return RawData[RunType](da)


def to_straws(da: RawData[RunType]) -> DataAsStraws[RunType]:
    return DataAsStraws[RunType](
        da.fold(
            dim='detector_id', sizes=dict(layer=4, tube=32, straw=7, pixel=512)
        ).flatten(dims=['layer', 'tube', 'straw'], to='straw')
    )


def detector_straw_mask(sample_straws: DataAsStraws[SampleRun]) -> DetectorStrawMask:
    return DetectorStrawMask(
        sample_straws.sum(['tof', 'pixel']).data < sc.scalar(300.0, unit='counts')
    )


def detector_beam_stop_mask(
    sample_straws: DataAsStraws[SampleRun],
) -> DetectorBeamStopMask:
    pos = sample_straws.coords['position']
    x = pos.fields.x
    y = pos.fields.y
    return DetectorBeamStopMask(
        (abs(x) < sc.scalar(0.03, unit='m')) & (abs(y) < sc.scalar(0.025, unit='m'))
    )


def detector_tube_edge_mask(
    sample_straws: DataAsStraws[SampleRun],
) -> DetectorTubeEdgeMask:
    return DetectorTubeEdgeMask(
        abs(sample_straws.coords['position'].fields.x) > sc.scalar(0.36, unit='m')
    )


def mask_detectors(
    da: DataAsStraws[RunType],
    straw_mask: Optional[DetectorStrawMask],
    beam_stop_mask: Optional[DetectorBeamStopMask],
    tube_edge_mask: Optional[DetectorTubeEdgeMask],
) -> MaskedData[RunType]:
    """Apply pixel-specific masks to raw data.

    Parameters
    ----------
    da:
        Raw data.
    edge_mask:
        Mask for detector edges.
    holder_mask:
        Mask for sample holder.
    """
    da = da.copy(deep=False)
    if straw_mask is not None:
        da.masks['bad_straws'] = straw_mask
    if beam_stop_mask is not None:
        da.masks['beam_stop'] = beam_stop_mask
    if tube_edge_mask is not None:
        da.masks['tube_edges'] = tube_edge_mask
    return MaskedData[RunType](da)


providers = [
    load_larmor_run,
    to_straws,
    detector_straw_mask,
    detector_beam_stop_mask,
    detector_tube_edge_mask,
    mask_detectors,
]
"""
Providers for direct beam
"""