# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import List

import numpy as np
import scipp as sc
from sciline import Pipeline

from .types import (
    BackgroundRun,
    BackgroundSubtractedIofQ,
    CleanMonitor,
    CleanSummedQ,
    DirectBeam,
    Incident,
    Denominator,
    Numerator,
    SampleRun,
    SolidAngle,
    TransmissionFraction,
    WavelengthBands,
    WavelengthBins,
)
from .i_of_q import resample_direct_beam


def _compute_efficiency_correction(
    iofq_full: sc.DataArray,
    iofq_bands: sc.DataArray,
    wavelength_band_dim: str,
    I0: sc.Variable,
) -> sc.DataArray:
    """
    Compute the factor by which to multiply the direct beam function inside each
    wavelength band so that the $I(Q)$ curves for the full wavelength range and inside
    the bands overlap.

    Parameters
    ----------
    iofq_full:
        The $I(Q)$ for the full wavelength range.
    iofq_bands:
        The $I(Q)$ for the wavelength bands.
    wavelength_band_dim:
        The name of the wavelength band dimension.
    I0:
        The intensity of the I(Q) for the known sample at the lowest Q value.
    """
    invalid = (iofq_bands.data <= sc.scalar(0.0)) | ~sc.isfinite(iofq_bands.data)
    data = np.where(invalid.values, np.nan, (iofq_bands.data / iofq_full.data).values)
    eff = np.nanmedian(data, axis=iofq_bands.dims.index('Q'))

    scaling = sc.values(iofq_full['Q', 0].data) / I0
    # Note: do not use a `set` here because the order of dimensions is important
    dims = [dim for dim in iofq_bands.dims if dim != 'Q']
    out = sc.array(dims=dims, values=eff) * scaling
    return out.rename_dims({wavelength_band_dim: 'wavelength'})


def direct_beam(pipeline: Pipeline, I0: sc.Variable, niter: int = 5) -> List[dict]:
    """
    Compute the direct beam function.

    Procedure:

    The idea behind the direct beam iterations is to determine an efficiency of the
    detectors as a function of wavelength.
    To calculate this, it is possible to compute $I(Q)$ for the full wavelength range,
    and for individual slices (bands) of the wavelength range.
    If the direct beam function used in the $I(Q)$ computation is correct, then $I(Q)$
    curves for the full wavelength range and inside the bands should overlap.

    We require two pipelines, one for the full wavelength range and one for the bands.

    The steps are as follows:

     1. Create a flat direct beam function, as a function of wavelength, with
        wavelength bins corresponding to the wavelength bands
     2. Calculate inside each band by how much one would have to multiply the final
        $I(Q)$ so that the curve would overlap with the full range curve
     3. Multiply the direct beam values inside each wavelength band by this factor
     4. Compare the full-range $I(Q)$ to a theoretical reference and add the
        corresponding additional scaling to the direct beam function
     5. Iterate a given number of times (typically less than 10) so as to gradually
        converge on a direct beam function

    TODO: For simplicity and robustness, we currently specify the number of times to
    iterate. We could imagine in the future having a convergence criterion instead to
    determine when to stop iterating.

    Parameters
    ----------
    pipeline:
        The pipeline to compute the differential scattering cross section I(Q).
    I0:
        The intensity of the I(Q) for the known sample at the lowest Q value.
    niter:
        The number of iterations to perform.
    """

    direct_beam_function = None
    bands = pipeline.compute(WavelengthBands)
    band_dim = (set(bands.dims) - {'wavelength'}).pop()

    full_wavelength_range = sc.concat([bands.min(), bands.max()], dim='wavelength')

    pipeline = pipeline.copy()
    # Append full wavelength range as extra band. This allows for running only a
    # single pipeline to compute both the I(Q) in bands and the I(Q) for the full
    # wavelength range.
    # pipeline[WavelengthBands] = sc.concat([bands, full_wavelength_range], dim=band_dim)
    wavelength_bins = pipeline.compute(WavelengthBins)
    pipeline[WavelengthBands] = wavelength_bins
    # bands = pipeline.compute(WavelengthBands)

    results = []

    # Compute checkpoints to avoid recomputing the same things in every iteration
    checkpoints = (
        TransmissionFraction[SampleRun],
        TransmissionFraction[BackgroundRun],
        SolidAngle[SampleRun],
        SolidAngle[BackgroundRun],
        CleanMonitor[SampleRun, Incident],
        CleanMonitor[BackgroundRun, Incident],
        CleanSummedQ[SampleRun, Numerator],
        CleanSummedQ[BackgroundRun, Numerator],
    )
    parts = (
        CleanSummedQ[SampleRun, Numerator],
        CleanSummedQ[BackgroundRun, Numerator],
        CleanSummedQ[SampleRun, Denominator],
        CleanSummedQ[BackgroundRun, Denominator],
    )

    for key, result in pipeline.compute(checkpoints).items():
        pipeline[key] = result

    # The first time we compute I(Q), the direct beam function is not in the
    # parameters, nor given by any providers, so it will be considered flat.
    # TODO: Should we have a check that DirectBeam cannot be computed from the
    # pipeline?
    parts = pipeline.compute(parts)
    # iofq0 = pipeline.compute(BackgroundSubtractedIofQ)
    # del iofq0.coords['wavelength']
    # iofq0 = iofq0.rename_dims({band_dim: 'wavelength'})
    # iofq0.coords['wavelength'] = wavelength_bins
    # print(iofq0)

    nom = parts[CleanSummedQ[SampleRun, Numerator]]
    denom0 = parts[CleanSummedQ[SampleRun, Denominator]].rename_dims(
        {band_dim: 'wavelength'}
    )
    del nom.coords['wavelength']
    nom = nom.rename_dims({band_dim: 'wavelength'})
    nom.coords['wavelength'] = sc.midpoints(wavelength_bins, dim='wavelength')
    denom0.coords['wavelength'] = wavelength_bins

    bnom = parts[CleanSummedQ[BackgroundRun, Numerator]]
    bdenom0 = parts[CleanSummedQ[BackgroundRun, Denominator]].rename_dims(
        {band_dim: 'wavelength'}
    )
    del bnom.coords['wavelength']
    bnom = bnom.rename_dims({band_dim: 'wavelength'})
    bnom.coords['wavelength'] = sc.midpoints(wavelength_bins, dim='wavelength')
    bdenom0.coords['wavelength'] = wavelength_bins

    print(denom0)
    print(f'{bands=}')
    for it in range(niter):
        print("Iteration", it)
        if direct_beam_function is not None:
            db = resample_direct_beam(
                direct_beam_function, wavelength_bins=wavelength_bins
            )
            denom = denom0 * db
            bdenom = bdenom0 * db
        else:
            denom = denom0.copy(deep=False)
            bdenom = bdenom0.copy(deep=False)
        iofq_full = nom.sum('wavelength') / denom.sum('wavelength') - bnom.sum(
            'wavelength'
        ) / bdenom.sum('wavelength')
        sections = []
        # tmp = iofq.copy(deep=False)
        if True:
            denom.coords['wavelength'] = sc.midpoints(wavelength_bins, dim='wavelength')
            bdenom.coords['wavelength'] = sc.midpoints(
                wavelength_bins, dim='wavelength'
            )
            nom.coords['wavelength'] = sc.midpoints(wavelength_bins, dim='wavelength')
            bnom.coords['wavelength'] = sc.midpoints(wavelength_bins, dim='wavelength')
        else:
            denom.coords['wavelength'] = wavelength_bins
            bdenom.coords['wavelength'] = wavelength_bins
            nom.coords['wavelength'] = wavelength_bins
            bnom.coords['wavelength'] = wavelength_bins
        for i in range(bands.sizes[band_dim]):
            bounds = bands[band_dim, i]
            band_num = nom['wavelength', bounds[0] : bounds[1]].sum('wavelength')
            band_denom = denom['wavelength', bounds[0] : bounds[1]].sum('wavelength')
            bband_num = bnom['wavelength', bounds[0] : bounds[1]].sum('wavelength')
            bband_denom = bdenom['wavelength', bounds[0] : bounds[1]].sum('wavelength')
            sections.append(band_num / band_denom - bband_num / bband_denom)
        iofq_bands = sc.concat(sections, dim=band_dim)
        iofq_bands.data = sc.nan_to_num(iofq_bands.data, nan=sc.scalar(0.0))
        print(iofq_bands['band', :10].values)
        iofq_bands.coords['wavelength'] = bands

        print(iofq_full)
        print(iofq_bands)
        # iofq_full = iofq['band', -1]
        # iofq_bands = iofq['band', :-1]

        if direct_beam_function is None:
            # Make a flat direct beam
            dims = [dim for dim in iofq_bands.dims if dim != 'Q']
            direct_beam_function = sc.DataArray(
                data=sc.ones(sizes={dim: iofq_bands.sizes[dim] for dim in dims}),
                coords={band_dim: sc.midpoints(bands, dim='wavelength').squeeze()},
            ).rename({band_dim: 'wavelength'})

        direct_beam_function *= _compute_efficiency_correction(
            iofq_full=iofq_full,
            iofq_bands=iofq_bands,
            wavelength_band_dim=band_dim,
            I0=I0,
        )

        # Insert new direct beam function into pipeline
        # pipeline[DirectBeam] = direct_beam_function

        results.append(
            {
                'iofq_full': iofq_full,
                'iofq_bands': iofq_bands,
                'direct_beam': direct_beam_function,
            }
        )
    return results
