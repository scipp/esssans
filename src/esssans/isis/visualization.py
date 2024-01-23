# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""
Plotting functions for ISIS data.
"""
from typing import Any

import scipp as sc


def plot_instrument(da: sc.DataArray, pixels_per_tube: int = 512, **kwargs: Any) -> Any:
    """
    Plot a 2-D instrument view of a data array.

    This is an alternative to the `scn.instrument_view` function, avoiding the 3-D
    rendering of the instrument. The exact X and Y coordinates of the pixels are
    used for the 2-D plot.

    Parameters
    ----------
    da:
        The data array to plot. Must have a 'position' coord and a single dimension.
    pixels_per_tube:
        The number of pixels per tube. Defaults to 512.
    kwargs:
        Additional arguments passed to `sc.plot`.
    """
    if da.bins is not None:
        da = da.hist()
    da.coords['x'] = da.coords['position'].fields.x.copy()
    da.coords['y'] = da.coords['position'].fields.y.copy()
    folded = da.fold(da.dim, sizes={'y': -1, 'x': pixels_per_tube})
    y = folded.coords['y']
    if sc.all(y.min('x') == y.max('x')):
        folded.coords['y'] = y.min('x')
    else:
        raise ValueError(
            'Cannot plot 2-D instrument view of data array with non-constant '
            'y coordinate along tubes. Use scippneutron.instrument_view instead.'
        )
    plot_kwargs = dict(aspect='equal', norm='log', figsize=(6, 10))
    plot_kwargs.update(kwargs)
    return folded.plot(**plot_kwargs)
