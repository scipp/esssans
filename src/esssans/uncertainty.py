# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Tools for handling statistical uncertainties."""

from typing import Dict, TypeVar, Union, overload

import numpy as np
import scipp as sc

T = TypeVar("T", bound=Union[sc.Variable, sc.DataArray])


@overload
def broadcast_with_upper_bound_variances(
    data: sc.Variable, template: sc.Variable
) -> sc.Variable:
    pass


@overload
def broadcast_with_upper_bound_variances(
    data: sc.DataArray, template: sc.DataArray
) -> sc.DataArray:
    pass


def broadcast_with_upper_bound_variances(
    data: Union[sc.Variable, sc.DataArray], template: Union[sc.Variable, sc.DataArray]
) -> Union[sc.Variable, sc.DataArray]:
    """
    Upper-bound estimate for errors from broadcasting a dense array to match the sizes
    of a given template array.

    We first count the number of non-masked elements in the template. To get the
    variances multiplier, we then divide that number by the volume of the dimensions of
    the input data array that are also found in the dimensions of the template. That
    scaling factor is used to multiply the variances of the input data array.
    Finally, an explicit broadcast is performed to bypass Scipp's safety check on
    broadcasting variances.

    Parameters
    ----------
    data
        The data array to be normalized.
    template
        The template data array.
    """
    if _no_variance_broadcast(data, template.sizes):
        return data
    data = data.copy()
    broadcast_sizes = {
        dim: size for dim, size in template.sizes.items() if dim not in data.dims
    }
    ones = sc.DataArray(data=sc.ones(sizes=broadcast_sizes))
    if isinstance(template, sc.DataArray):
        broadcast_dims = set(broadcast_sizes)
        ones.masks.update(
            {
                key: mask
                for key, mask in template.masks.items()
                if set(mask.dims).issubset(broadcast_dims)
            }
        )
    # mult = ones.sum()
    # div = np.prod([data.sizes[dim] for dim in data.dims if dim in template.dims])
    # data.variances *= (mult / div).value
    data.variances *= ones.sum().value
    return data.broadcast(sizes={**template.sizes, **data.sizes}).copy()


def drop_variances_if_broadcast(
    data: Union[sc.Variable, sc.DataArray], template: Union[sc.Variable, sc.DataArray]
) -> Union[sc.Variable, sc.DataArray]:
    if _no_variance_broadcast(data, template.sizes):
        return data
    return sc.values(data)


def _no_variance_broadcast(
    data: Union[sc.Variable, sc.DataArray], sizes: Dict[str, int]
) -> bool:
    return (data.variances is None) or all(
        data.sizes.get(dim) == size for dim, size in sizes.items()
    )


def broadcast_to_events_with_upper_bound_variances(
    da: sc.DataArray, *, events: sc.DataArray
) -> sc.DataArray:
    """
    Upper-bound estimate for errors from normalization in event-mode.

    Count the number of events in each bin of the input data array. Then scale the
    variances by the number of events in each bin. An explicit broadcast is performed
    to bypass Scipp's safety check on broadcasting variances.

    Details will be published in an upcoming publication by Simon Heybrock et al.
    """
    if da.variances is None:
        return da
    if da.sizes != events.sizes:
        # This is a safety check, but we should never get here.
        raise ValueError(f"Sizes {da.sizes} do not match event sizes {events.sizes}")
    # Given how this function is used currently (in the context of normalization
    # with matching binning in numerator and denominator, not using scipp.lookup),
    # we can simply count the events in the existing binning.
    da.variances *= events.bins.size().values
    da.data = sc.bins_like(events, da.data)
    return da
