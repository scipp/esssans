# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import uuid

import scipp as sc
from ess.reduce.uncertainty import UncertaintyBroadcastMode, broadcast_uncertainties
from scipp.scipy.interpolate import interp1d

from .common import mask_range
from .logging import get_logger
from .types import (
    BackgroundRun,
    BackgroundSubtractedIofQ,
    BackgroundSubtractedIofQxy,
    CleanDirectBeam,
    CleanMonitor,
    CleanQ,
    CleanQxy,
    CleanSummedQ,
    CleanSummedQxy,
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
    return CleanMonitor(monitor)


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


def bin_in_q(
    data: CleanQ[ScatteringRunType, IofQPart],
    q_bins: QBins,
    dims_to_keep: DimsToKeep,
) -> CleanSummedQ[ScatteringRunType, IofQPart]:
    """
    Merges data from all pixels into a single I(Q) spectrum:

    * In the case of event data, events in all bins are concatenated
    * In the case of dense data, counts in all spectra are summed

    Parameters
    ----------
    data:
        A DataArray containing the data that is to be converted to Q.
    q_bins:
        The binning in Q to be used.
    dims_to_keep:
        Dimensions that should not be reduced and thus still be present in the final
        I(Q) result (this is typically the layer dimension).

    Returns
    -------
    :
        The input data converted to Q and then summed over all detector pixels.
    """
    out = _bin_in_q(data=data, edges={'Q': q_bins}, dims_to_keep=dims_to_keep)
    return CleanSummedQ[ScatteringRunType, IofQPart](out)


def bin_in_qxy(
    data: CleanQxy[ScatteringRunType, IofQPart],
    qx_bins: QxBins,
    qy_bins: QyBins,
    dims_to_keep: DimsToKeep,
) -> CleanSummedQxy[ScatteringRunType, IofQPart]:
    """
    Merges data from all pixels into a single I(Q) spectrum:

    * In the case of event data, events in all bins are concatenated
    * In the case of dense data, counts in all spectra are summed

    Parameters
    ----------
    data:
        A DataArray containing the data that is to be converted to Q.
    qx_bins:
        The binning in Qx to be used.
    qy_bins:
        The binning in Qy to be used.
    dims_to_keep:
        Dimensions that should not be reduced and thus still be present in the final
        I(Q) result (this is typically the layer dimension).

    Returns
    -------
    :
        The input data converted to Qx and Qy and then summed over all detector pixels.
    """
    # We make Qx the inner dim, such that plots naturally show Qx on the x-axis.
    out = _bin_in_q(
        data=data,
        edges={'Qy': qy_bins, 'Qx': qx_bins},
        dims_to_keep=dims_to_keep,
    )
    return CleanSummedQxy[ScatteringRunType, IofQPart](out)


def _bin_in_q(
    data: sc.DataArray, edges: dict[str, sc.Variable], dims_to_keep: tuple[str, ...]
) -> sc.DataArray:
    dims_to_reduce = set(data.dims) - {'wavelength'}
    if dims_to_keep is not None:
        dims_to_reduce -= set(dims_to_keep)

    if data.bins is not None:
        q_all_pixels = data.bins.concat(dims_to_reduce)
        # q_all_pixels may just have a single bin now, which currently yields
        # inferior performance when binning (no/bad multi-threading?).
        # We operate on the content buffer for better multi-threaded performance.
        if q_all_pixels.ndim == 0:
            content = q_all_pixels.bins.constituents['data']
            out = content.bin(**edges).assign_coords(q_all_pixels.coords)
        else:
            out = q_all_pixels.bin(**edges)
    else:
        # We want to flatten data to make histogramming cheaper (avoiding allocation of
        # large output before summing). We strip unnecessary content since it makes
        # flattening more expensive.
        stripped = data.copy(deep=False)
        for name, coord in data.coords.items():
            if (
                name not in {'Q', 'Qx', 'Qy', 'wavelength'}
                and set(coord.dims) & dims_to_reduce
            ):
                del stripped.coords[name]
        to_flatten = [dim for dim in data.dims if dim in dims_to_reduce]

        # Make dims to flatten contiguous, keep wavelength as the last dim
        data_dims = list(stripped.dims)
        for dim in [*to_flatten, 'wavelength']:
            data_dims.remove(dim)
            data_dims.append(dim)
        stripped = stripped.transpose(data_dims)
        # Flatten to helper dim such that `hist` will turn this into the new Q dim(s).
        # For sc.hist this has to be named 'Q'.
        helper_dim = 'Q'
        flat = stripped.flatten(dims=to_flatten, to=helper_dim)

        if len(edges) == 1:
            out = flat.hist(**edges)
        else:
            # sc.hist (or the underlying sc.bin) cannot deal with extra data dims,
            # work around by flattening and regrouping.
            for dim in flat.dims:
                if dim == helper_dim:
                    continue
                if dim not in flat.coords:
                    flat.coords[dim] = sc.arange(dim, flat.sizes[dim])
            out = (
                flat.flatten(to=str(uuid.uuid4()))
                .group(*[flat.coords[dim] for dim in flat.dims if dim != helper_dim])
                .drop_coords(dims_to_keep or ())
                .hist(**edges)
            )
    return out.squeeze()


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
    bin_in_q,
    bin_in_qxy,
    subtract_background,
    subtract_background_xy,
)
