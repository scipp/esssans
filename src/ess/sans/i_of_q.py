# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.scipy.interpolate import interp1d

from ess.reduce.uncertainty import UncertaintyBroadcastMode, broadcast_uncertainties

from .common import mask_range
from .logging import get_logger
from .types import (
    BackgroundRun,
    BackgroundSubtractedIofQ,
    BackgroundSubtractedIofQxy,
    CleanDirectBeam,
    CorrectedMonitor,
    CorrectedQ,
    CorrectedQxy,
    DimsToKeep,
    DirectBeam,
    IofQ,
    IofQPart,
    IofQxy,
    MonitorType,
    NonBackgroundWavelengthRange,
    QBins,
    QxBins,
    QyBins,
    ReturnEvents,
    RunType,
    SampleRun,
    ScatteringRunType,
    WavelengthBins,
    WavelengthMonitor,
)


def preprocess_monitor_data(
    monitor: WavelengthMonitor[RunType, MonitorType],
    wavelength_bins: WavelengthBins,
    non_background_range: NonBackgroundWavelengthRange,
    uncertainties: UncertaintyBroadcastMode,
) -> CorrectedMonitor[RunType, MonitorType]:
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
        :py:class:`ess.reduce.uncertainty.UncertaintyBroadcastMode` for details.

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
        monitor -= broadcast_uncertainties(
            background, prototype=monitor, mode=uncertainties
        )
    return CorrectedMonitor(monitor)


def resample_direct_beam(
    direct_beam: DirectBeam, wavelength_bins: WavelengthBins
) -> CleanDirectBeam:
    """
    If the wavelength binning of the direct beam function does not match the requested
    ``wavelength_bins``, perform a 1d interpolation of the function onto the bins.

    Parameters
    ----------
    direct_beam:
        The DataArray containing the direct beam function (it should have a dimension
        of wavelength).
    wavelength_bins:
        The binning in wavelength that the direct beam function should be resampled to.

    Returns
    -------
    :
        The direct beam function resampled to the requested resolution.
    """
    if direct_beam is None:
        return CleanDirectBeam(
            sc.DataArray(
                sc.ones(dims=wavelength_bins.dims, shape=[len(wavelength_bins) - 1]),
                coords={'wavelength': wavelength_bins},
            )
        )
    if sc.identical(direct_beam.coords['wavelength'], wavelength_bins):
        return direct_beam
    if direct_beam.variances is not None:
        logger = get_logger('sans')
        logger.warning(
            'An interpolation is being performed on the direct_beam function. '
            'The variances in the direct_beam function will be dropped.'
        )
    func = interp1d(
        sc.values(direct_beam),
        'wavelength',
        fill_value="extrapolate",
        bounds_error=False,
    )
    return CleanDirectBeam(func(wavelength_bins, midpoints=True))


def _subtract_background(
    sample: sc.DataArray,
    background: sc.DataArray,
    return_events: ReturnEvents,
) -> sc.DataArray:
    if return_events and sample.bins is not None and background.bins is not None:
        return sample.bins.concatenate(-background)
    if sample.bins is not None:
        sample = sample.bins.sum()
    if background.bins is not None:
        background = background.bins.sum()
    return sample - background


def subtract_background(
    sample: IofQ[SampleRun],
    background: IofQ[BackgroundRun],
    return_events: ReturnEvents,
) -> BackgroundSubtractedIofQ:
    return BackgroundSubtractedIofQ(
        _subtract_background(
            sample=sample, background=background, return_events=return_events
        )
    )


def subtract_background_xy(
    sample: IofQxy[SampleRun],
    background: IofQxy[BackgroundRun],
    return_events: ReturnEvents,
) -> BackgroundSubtractedIofQxy:
    return BackgroundSubtractedIofQxy(
        _subtract_background(
            sample=sample, background=background, return_events=return_events
        )
    )


providers = (
    preprocess_monitor_data,
    resample_direct_beam,
    subtract_background,
    subtract_background_xy,
)
