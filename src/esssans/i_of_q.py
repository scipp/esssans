# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import uuid
from typing import Optional

import scipp as sc
from .types import (
    BackgroundRun,
    BackgroundSubtractedIofQ,
    CleanQ,
    CleanSummedQ,
    DimsToKeep,
    IofQ,
    IofQPart,
    QBins,
    QxyBins,
    ReturnEvents,
    SampleRun,
    ScatteringRunType,
)


def bin_in_q(
    data: CleanQ[ScatteringRunType, IofQPart],
    q_bins: Optional[QBins],
    qxy_bins: Optional[QxyBins],
    dims_to_keep: Optional[DimsToKeep],
) -> CleanSummedQ[ScatteringRunType, IofQPart]:
    """
    Merges all spectra:

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
    dims_to_reduce = set(data.dims) - {'wavelength'}
    if dims_to_keep is not None:
        dims_to_reduce -= set(dims_to_keep)

    if qxy_bins:
        # We make Qx the inner dim, such that plots naturally show Qx on the x-axis.
        edges = {'Qy': qxy_bins['Qy'], 'Qx': qxy_bins['Qx']}
    else:
        edges = {'Q': q_bins}

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
        for dim in to_flatten + ['wavelength']:
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
    return CleanSummedQ[ScatteringRunType, IofQPart](out.squeeze())


def subtract_background(
    sample: IofQ[SampleRun],
    background: IofQ[BackgroundRun],
    return_events: ReturnEvents,
) -> BackgroundSubtractedIofQ:
    if return_events and sample.bins is not None and background.bins is not None:
        return sample.bins.concatenate(-background)
    if sample.bins is not None:
        sample = sample.bins.sum()
    if background.bins is not None:
        background = background.bins.sum()
    return BackgroundSubtractedIofQ(sample - background)


providers = (bin_in_q, subtract_background)
