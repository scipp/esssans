# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest
import sciline
import scipp as sc
from ess import isissans as isis
from ess import sans
from ess.isissans import MonitorOffset, SampleOffset, sans2d
from ess.sans.types import (
    BackgroundRun,
    BackgroundSubtractedIofQ,
    BeamCenter,
    CorrectForGravity,
    DimsToKeep,
    DirectBeam,
    DirectBeamFilename,
    EmptyBeamRun,
    Filename,
    Incident,
    IofQ,
    MaskedData,
    NeXusMonitorName,
    NonBackgroundWavelengthRange,
    QBins,
    RawDetector,
    ReturnEvents,
    SampleRun,
    SolidAngle,
    Transmission,
    UncertaintyBroadcastMode,
    WavelengthBands,
    WavelengthBins,
    WavelengthMask,
)


def make_params() -> dict:
    params = isis.default_parameters()
    params[WavelengthBins] = sc.linspace(
        'wavelength', start=2.0, stop=16.0, num=141, unit='angstrom'
    )
    params[WavelengthMask] = sc.DataArray(
        data=sc.array(dims=['wavelength'], values=[True]),
        coords={
            'wavelength': sc.array(
                dims=['wavelength'], values=[2.21, 2.59], unit='angstrom'
            )
        },
    )
    params[sans2d.LowCountThreshold] = sc.scalar(100.0, unit='counts')

    params[QBins] = sc.linspace(
        dim='Q', start=0.01, stop=0.55, num=141, unit='1/angstrom'
    )
    params[DirectBeamFilename] = isis.data.sans2d_tutorial_direct_beam()
    params[Filename[SampleRun]] = isis.data.sans2d_tutorial_sample_run()
    params[Filename[BackgroundRun]] = isis.data.sans2d_tutorial_background_run()
    params[Filename[EmptyBeamRun]] = isis.data.sans2d_tutorial_empty_beam_run()

    params[NeXusMonitorName[Incident]] = 'monitor2'
    params[NeXusMonitorName[Transmission]] = 'monitor4'
    params[SampleOffset] = sc.vector([0.0, 0.0, 0.053], unit='m')
    params[MonitorOffset[Transmission]] = sc.vector([0.0, 0.0, -6.719], unit='m')

    params[NonBackgroundWavelengthRange] = sc.array(
        dims=['wavelength'], values=[0.7, 17.1], unit='angstrom'
    )
    params[CorrectForGravity] = True
    params[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
    params[ReturnEvents] = False
    params[DimsToKeep] = ()
    params[BeamCenter] = sc.vector([0, 0, 0], unit='m')

    return params


def sans2d_providers():
    return list(
        sans.providers
        + isis.providers
        + isis.sans2d.providers
        + isis.mantidio.providers
        + (
            isis.data.load_tutorial_direct_beam,
            isis.data.load_tutorial_run,
            isis.data.transmission_from_background_run,
            isis.data.transmission_from_sample_run,
        )
    )


def test_can_create_pipeline():
    sciline.Pipeline(sans2d_providers(), params=make_params())


@pytest.mark.parametrize(
    'uncertainties',
    [UncertaintyBroadcastMode.drop, UncertaintyBroadcastMode.upper_bound],
)
def test_pipeline_can_compute_background_subtracted_IofQ(uncertainties):
    params = make_params()
    params[UncertaintyBroadcastMode] = uncertainties
    pipeline = sciline.Pipeline(sans2d_providers(), params=params)
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)


def test_pipeline_can_compute_background_subtracted_IofQ_in_wavelength_bands():
    params = make_params()
    params[WavelengthBands] = sc.linspace(
        'wavelength', start=2.0, stop=16.0, num=11, unit='angstrom'
    )
    pipeline = sciline.Pipeline(sans2d_providers(), params=params)
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('band', 'Q')
    assert result.sizes['band'] == 10


def test_pipeline_wavelength_bands_is_optional():
    params = make_params()
    pipeline = sciline.Pipeline(sans2d_providers(), params=params)
    pipeline[BeamCenter] = sans.beam_center_from_center_of_mass(pipeline)
    noband = pipeline.compute(BackgroundSubtractedIofQ)
    assert pipeline.compute(WavelengthBands) is None
    band = sc.linspace('wavelength', 2.0, 16.0, num=2, unit='angstrom')
    pipeline[WavelengthBands] = band
    assert sc.identical(band, pipeline.compute(WavelengthBands))
    withband = pipeline.compute(BackgroundSubtractedIofQ)
    assert sc.identical(noband, withband)


def test_workflow_is_deterministic():
    params = make_params()
    params[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.drop
    pipeline = sciline.Pipeline(sans2d_providers(), params=params)
    pipeline[BeamCenter] = sans.beam_center_from_center_of_mass(pipeline)
    # This is Sciline's default scheduler, but we want to be explicit here
    scheduler = sciline.scheduler.DaskScheduler()
    graph = pipeline.get(IofQ[SampleRun], scheduler=scheduler)
    reference = graph.compute().data
    result = graph.compute().data
    assert sc.identical(sc.values(result), sc.values(reference))


def test_pipeline_raises_VariancesError_if_normalization_errors_not_dropped():
    params = make_params()
    params[NonBackgroundWavelengthRange] = (
        None  # Make sure we raise in iofq_denominator
    )
    params[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.fail
    pipeline = sciline.Pipeline(sans2d_providers(), params=params)
    with pytest.raises(sc.VariancesError):
        pipeline.compute(BackgroundSubtractedIofQ)


def test_uncertainty_broadcast_mode_drop_yields_smaller_variances():
    params = make_params()
    # Errors with the full range have some NaNs or infs
    params[QBins] = sc.linspace(
        dim='Q', start=0.01, stop=0.5, num=141, unit='1/angstrom'
    )
    params[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.drop
    pipeline = sciline.Pipeline(sans2d_providers(), params=params)
    drop = pipeline.compute(IofQ[SampleRun]).data
    params[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
    pipeline = sciline.Pipeline(sans2d_providers(), params=params)
    upper_bound = pipeline.compute(IofQ[SampleRun]).data
    assert sc.all(sc.variances(drop) < sc.variances(upper_bound)).value


def test_pipeline_can_visualize_background_subtracted_IofQ():
    pipeline = sciline.Pipeline(sans2d_providers(), params=make_params())
    pipeline.visualize(BackgroundSubtractedIofQ)


def test_pipeline_can_compute_intermediate_results():
    pipeline = sciline.Pipeline(sans2d_providers(), params=make_params())
    result = pipeline.compute(SolidAngle[SampleRun])
    assert result.dims == ('spectrum',)


def pixel_dependent_direct_beam(
    filename: DirectBeamFilename, shape: RawDetector[SampleRun]
) -> DirectBeam:
    direct_beam = isis.data.load_tutorial_direct_beam(filename)
    sizes = {'spectrum': shape.sizes['spectrum'], **direct_beam.sizes}
    return DirectBeam(direct_beam.broadcast(sizes=sizes).copy())


@pytest.mark.parametrize(
    'uncertainties',
    [UncertaintyBroadcastMode.drop, UncertaintyBroadcastMode.upper_bound],
)
def test_pixel_dependent_direct_beam_is_supported(uncertainties):
    params = make_params()
    params[UncertaintyBroadcastMode] = uncertainties
    pipeline = sciline.Pipeline(sans2d_providers(), params=params)
    pipeline.insert(pixel_dependent_direct_beam)
    pipeline[BeamCenter] = sc.vector([0, 0, 0], unit='m')
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)


MANTID_BEAM_CENTER = sc.vector([0.09288, -0.08195, 0], unit='m')


def test_beam_center_from_center_of_mass_is_close_to_verified_result():
    params = make_params()
    providers = sans2d_providers()
    pipeline = sciline.Pipeline(providers, params=params)
    center = sans.beam_center_from_center_of_mass(pipeline)
    # This is the result obtained from Mantid, using the full IofQ
    # calculation. The difference is about 3 mm in X or Y, probably due to a bias
    # introduced by the sample holder, which the center-of-mass approach cannot ignore.
    assert sc.allclose(center, MANTID_BEAM_CENTER, atol=sc.scalar(3e-3, unit='m'))


def test_beam_center_from_center_of_mass_independent_of_set_beam_center():
    params = make_params()
    providers = sans2d_providers()
    pipeline = sciline.Pipeline(providers, params=params)
    pipeline[BeamCenter] = sc.vector([0.1, -0.1, 0], unit='m')
    center = sans.beam_center_from_center_of_mass(pipeline)
    assert sc.allclose(center, MANTID_BEAM_CENTER, atol=sc.scalar(3e-3, unit='m'))


def test_beam_center_finder_without_direct_beam_reproduces_verified_result():
    params = make_params()
    del params[DirectBeamFilename]
    providers = sans2d_providers()
    pipeline = sciline.Pipeline(providers, params=params)
    pipeline[DirectBeam] = None
    center = sans.beam_center_finder.beam_center_from_iofq(
        workflow=pipeline, q_bins=sc.linspace('Q', 0.02, 0.3, 71, unit='1/angstrom')
    )
    assert sc.allclose(center, MANTID_BEAM_CENTER, atol=sc.scalar(2e-3, unit='m'))


def test_beam_center_can_get_closer_to_verified_result_with_low_counts_mask():
    def low_counts_mask(
        sample: RawDetector[SampleRun],
        low_counts_threshold: sans2d.LowCountThreshold,
    ) -> sans2d.SampleHolderMask:
        return sans2d.SampleHolderMask(sample.hist().data < low_counts_threshold)

    params = make_params()
    params[sans2d.LowCountThreshold] = sc.scalar(80.0, unit='counts')
    del params[DirectBeamFilename]
    providers = sans2d_providers()
    providers.remove(sans2d.sample_holder_mask)
    providers.append(low_counts_mask)
    pipeline = sciline.Pipeline(providers, params=params)
    pipeline[DirectBeam] = None
    q_bins = sc.linspace('Q', 0.02, 0.3, 71, unit='1/angstrom')
    center = sans.beam_center_finder.beam_center_from_iofq(
        workflow=pipeline, q_bins=q_bins
    )
    assert sc.allclose(center, MANTID_BEAM_CENTER, atol=sc.scalar(5e-4, unit='m'))


def test_beam_center_finder_works_with_direct_beam():
    params = make_params()
    providers = sans2d_providers()
    pipeline = sciline.Pipeline(providers, params=params)
    q_bins = sc.linspace('Q', 0.02, 0.3, 71, unit='1/angstrom')
    center_with_direct_beam = sans.beam_center_finder.beam_center_from_iofq(
        workflow=pipeline, q_bins=q_bins
    )
    assert sc.allclose(
        center_with_direct_beam, MANTID_BEAM_CENTER, atol=sc.scalar(2e-3, unit='m')
    )


def test_beam_center_finder_independent_of_set_beam_center():
    params = make_params()
    providers = sans2d_providers()
    pipeline = sciline.Pipeline(providers, params=params)
    pipeline[BeamCenter] = sc.vector([0.1, -0.1, 0], unit='m')
    q_bins = sc.linspace('Q', 0.02, 0.3, 71, unit='1/angstrom')
    center_with_direct_beam = sans.beam_center_finder.beam_center_from_iofq(
        workflow=pipeline, q_bins=q_bins
    )
    assert sc.allclose(
        center_with_direct_beam, MANTID_BEAM_CENTER, atol=sc.scalar(2e-3, unit='m')
    )


def test_beam_center_finder_works_with_pixel_dependent_direct_beam():
    q_bins = sc.linspace('Q', 0.02, 0.3, 71, unit='1/angstrom')
    params = make_params()
    providers = sans2d_providers()
    pipeline = sciline.Pipeline(providers, params=params)
    center_pixel_independent_direct_beam = (
        sans.beam_center_finder.beam_center_from_iofq(workflow=pipeline, q_bins=q_bins)
    )

    direct_beam = pipeline.compute(DirectBeam)
    pixel_dependent_direct_beam = direct_beam.broadcast(
        sizes={
            'spectrum': pipeline.compute(MaskedData[SampleRun]).sizes['spectrum'],
            'wavelength': direct_beam.sizes['wavelength'],
        }
    ).copy()

    providers = sans2d_providers()
    pipeline = sciline.Pipeline(providers, params=params)
    pipeline[DirectBeam] = pixel_dependent_direct_beam

    center = sans.beam_center_finder.beam_center_from_iofq(
        workflow=pipeline, q_bins=q_bins
    )
    assert sc.identical(center, center_pixel_independent_direct_beam)


def test_workflow_runs_without_gravity_if_beam_center_is_provided():
    params = make_params()
    params[CorrectForGravity] = False
    pipeline = sciline.Pipeline(sans2d_providers(), params=params)
    da = pipeline.compute(RawDetector[SampleRun])
    del da.coords['gravity']
    pipeline[RawDetector[SampleRun]] = da
    pipeline[BeamCenter] = MANTID_BEAM_CENTER
    result = pipeline.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)
