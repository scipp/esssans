# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
This modules defines the domain types uses in esssans.

The domain types are used to define parameters and to request results from a Sciline
pipeline."""
from enum import Enum
from typing import NewType, TypeVar

import sciline
import scipp as sc

# 1  TypeVars used to parametrize the generic parts of the workflow

# 1.1  Run types
BackgroundRun = NewType('BackgroundRun', int)
"""Background run"""
EmptySampleHolderRun = NewType('EmptySampleHolderRun', int)
"""Run where the sample holder was empty (sometimes called 'direct run')"""
SampleRun = NewType('SampleRun', int)
"""Sample run"""
SampleTransmissionRun = NewType('SampleTransmissionRun', int)
"""Sample run for measuring transmission with monitors"""
RunType = TypeVar(
    'RunType', BackgroundRun, EmptySampleHolderRun, SampleRun, SampleTransmissionRun
)
"""
TypeVar used for specifying BackgroundRun, EmptySampleHolderRun, SampleRun, or
SampleTransmissionRun.
"""

# 1.2  Monitor types
# SampleIncidentMonitor = NewType('SampleIncidentMonitor', int)
# """Sample incident monitor"""
# SampleTransmissionMonitor = NewType('SampleTransmissionMonitor', int)
# """Sample transmission monitor"""
# EmptySampleHolderIncidentMonitor = NewType('EmptySampleHolderIncidentMonitor', int)
# """Empty sample holder incident monitor"""
# EmptySampleHolderTransmissionMonitor = NewType(
#     'EmptySampleHolderTransmissionMonitor', int
# )
# """Empty sample holder transmission monitor"""
# MonitorType = TypeVar(
#     'MonitorType',
#     SampleIncidentMonitor,
#     SampleTransmissionMonitor,
#     EmptySampleHolderIncidentMonitor,
#     EmptySampleHolderTransmissionMonitor,
# )
# """TypeVar used for specifying Incident or Transmission monitor type"""

Incident = NewType('Incident', int)
"""Incident monitor"""
Transmission = NewType('Transmission', int)
"""Transmission monitor"""
MonitorType = TypeVar('MonitorType', Incident, Transmission)
"""TypeVar used for specifying Incident or Transmission monitor type"""

# 1.3  Numerator and denominator of IofQ
Numerator = NewType('Numerator', sc.DataArray)
"""Numerator of IofQ"""
Denominator = NewType('Denominator', sc.DataArray)
"""Denominator of IofQ"""
IofQPart = TypeVar('IofQPart', Numerator, Denominator)
"""TypeVar used for specifying Numerator or Denominator of IofQ"""

# 2  Workflow parameters

UncertaintyBroadcastMode = Enum(
    'UncertaintyBroadcastMode', ['drop', 'upper_bound', 'fail']
)
"""
Mode for broadcasting uncertainties.

See https://doi.org/10.3233/JNR-220049 for context.
"""

WavelengthBins = NewType('WavelengthBins', sc.Variable)
"""Wavelength binning"""

WavelengthBands = NewType('WavelengthBands', sc.Variable)
"""Wavelength bands. Typically a single band, set to first and last value of
WavelengthBins."""

QBins = NewType('QBins', sc.Variable)
"""Q binning"""

NonBackgroundWavelengthRange = NewType('NonBackgroundWavelengthRange', sc.Variable)
"""Range of wavelengths that are not considered background in the monitor"""

DirectBeamFilename = NewType('DirectBeamFilename', str)
"""Filename of direct beam correction"""

BeamCenter = NewType('BeamCenter', sc.Variable)
"""Beam center, may be set directly or computed using beam-center finder"""

WavelengthMask = NewType('WavelengthMask', sc.DataArray)
"""Optional wavelength mask"""

CorrectForGravity = NewType('CorrectForGravity', bool)
"""Whether to correct for gravity when computing wavelength and Q."""


class NeXusMonitorName(sciline.Scope[MonitorType, str], str):
    """Name of Incident|Transmission monitor in NeXus file"""


class Filename(sciline.Scope[RunType, str], str):
    """Filename of BackgroundRun|EmptySampleHolderRun|SampleRun"""


# 3  Workflow (intermediate) results

DetectorEdgeMask = NewType('DetectorEdgeMask', sc.Variable)
"""Detector edge mask"""
SampleHolderMask = NewType('SampleHolderMask', sc.Variable)
"""Sample holder mask"""

DirectBeam = NewType('DirectBeam', sc.DataArray)
"""Direct beam"""

TransmissionFraction = NewType('TransmissionFraction', sc.DataArray)
"""
Transmission fraction for inspection purposes.

The IofQ computation does not use this, it uses the monitors directly.
"""


CleanDirectBeam = NewType('CleanDirectBeam', sc.DataArray)
"""Direct beam after resampling to required wavelength bins"""


class SolidAngle(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Solid angle of detector pixels seen from sample position"""


class RawData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Raw data"""


class MaskedData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Raw data with pixel-specific masks applied"""


class CalibratedMaskedData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Raw data with pixel-specific masks applied and calibrated pixel positions"""


class NormWavelengthTerm(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Normalization term (numerator) for IofQ before scaling with solid-angle."""


class Clean(sciline.ScopeTwoParams[RunType, IofQPart, sc.DataArray], sc.DataArray):
    """
    Prerequisite for IofQ numerator or denominator.

    This can either be the sample or background counts, converted to wavelength,
    or the respective normalization terms computed from the respective solid angle,
    direct beam, and monitors.
    """


class CleanMasked(
    sciline.ScopeTwoParams[RunType, IofQPart, sc.DataArray], sc.DataArray
):
    """Result of applying wavelength masking to :py:class:`Clean`"""


class CleanQ(sciline.ScopeTwoParams[RunType, IofQPart, sc.DataArray], sc.DataArray):
    """Result of converting :py:class:`CleanMasked` to Q"""


class CleanSummedQ(
    sciline.ScopeTwoParams[RunType, IofQPart, sc.DataArray], sc.DataArray
):
    """Result of histogramming/binning :py:class:`CleanQ` over all pixels into Q bins"""


class IofQ(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """I(Q)"""


BackgroundSubtractedIofQ = NewType('BackgroundSubtractedIofQ', sc.DataArray)
"""I(Q) with background (given by I(Q) of the background run) subtracted"""


class RawMonitor(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """Raw monitor data"""


class WavelengthMonitor(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """Monitor data converted to wavelength"""


class CleanMonitor(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """Monitor data cleaned of background counts"""


# 4  Direct beam iterations

DirectBeamNumberOfSamplingPoints = NewType('DirectBeamNumberOfSamplingPoints', int)
"""Number of points to sample the direct beam with"""

DirectBeamSamplingWavelengthWidth = NewType(
    'DirectBeamSamplingWavelengthWidth', sc.Variable
)
"""Width in wavelength of the sampling points for the direct beam"""

DirectBeamWavelengthSamplingPoints = NewType(
    'DirectBeamWavelengthSamplingPoints', sc.Variable
)
"""Locations in wavelength where the direct beam is sampled"""

SourcePosition = NewType('SourcePosition', sc.Variable)
"""The position of the source"""


class PatchedMonitor(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """Patched monitor data"""
