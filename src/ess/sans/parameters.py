# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 202 Scipp contributors (https://github.com/scipp)
"""
Default parameters, providers and utility functions for the loki workflow.
"""

from __future__ import annotations

from typing import Any

import scipp as sc
from ess.reduce.parameter import (
    BinEdgesParameter,
    BooleanParameter,
    FilenameParameter,
    MultiFilenameParameter,
    Parameter,
    ParamWithOptions,
    StringParameter,
    VectorParameter,
)
from sciline.typing import Key

from ..sans.types import (
    BackgroundRun,
    BeamCenter,
    CorrectForGravity,
    DirectBeam,
    DirectBeamFilename,
    EmptyBeamRun,
    Filename,
    Incident,
    NeXusDetectorName,
    NeXusMonitorName,
    PixelMaskFilename,
    PixelShapePath,
    QBins,
    ReturnEvents,
    SampleRun,
    TransformationPath,
    Transmission,
    TransmissionRun,
    UncertaintyBroadcastMode,
    WavelengthBins,
)


def make_parameter_mapping(*, defaults: dict[Key, Any]) -> dict[Key, Parameter]:
    """
    Return mapping of keys to parameters.
    """
    return {
        CorrectForGravity: BooleanParameter.from_type(
            CorrectForGravity, default=defaults.get(CorrectForGravity, False)
        ),
        NeXusDetectorName: StringParameter.from_type(NeXusDetectorName),
        NeXusMonitorName[Incident]: StringParameter.from_type(
            NeXusMonitorName[Incident], default=defaults[NeXusMonitorName[Incident]]
        ),
        NeXusMonitorName[Transmission]: StringParameter.from_type(
            NeXusMonitorName[Transmission],
            default=defaults[NeXusMonitorName[Transmission]],
        ),
        TransformationPath: StringParameter.from_type(
            TransformationPath, default=defaults[TransformationPath]
        ),
        PixelMaskFilename: MultiFilenameParameter.from_type(PixelMaskFilename),
        PixelShapePath: StringParameter.from_type(
            PixelShapePath, default=defaults[PixelShapePath]
        ),
        # [more default params]
        # Should this be ReductionMode (EventMode/HistogramMode)?
        ReturnEvents: BooleanParameter.from_type(ReturnEvents, default=False),
        UncertaintyBroadcastMode: ParamWithOptions.from_enum(
            UncertaintyBroadcastMode, default=UncertaintyBroadcastMode.upper_bound
        ),
        Filename[SampleRun]: FilenameParameter.from_type(Filename[SampleRun]),
        Filename[BackgroundRun]: FilenameParameter.from_type(Filename[BackgroundRun]),
        Filename[TransmissionRun[SampleRun]]: FilenameParameter.from_type(
            Filename[TransmissionRun[SampleRun]]
        ),
        Filename[TransmissionRun[BackgroundRun]]: FilenameParameter.from_type(
            Filename[TransmissionRun[BackgroundRun]]
        ),
        Filename[EmptyBeamRun]: FilenameParameter.from_type(Filename[EmptyBeamRun]),
        WavelengthBins: BinEdgesParameter(
            WavelengthBins, dim='wavelength', unit='angstrom'
        ),
        QBins: BinEdgesParameter(QBins, dim='Q', unit='1/angstrom'),
        DirectBeam: StringParameter.from_type(
            DirectBeam, switchable=True, optional=True, default=None
        ),
        DirectBeamFilename: FilenameParameter.from_type(
            DirectBeamFilename, switchable=True
        ),
        BeamCenter: VectorParameter.from_type(
            BeamCenter, default=sc.vector([0, 0, 0], unit='m')
        ),
    }
