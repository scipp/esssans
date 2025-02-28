# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from pathlib import Path

from ess.sans.data import Registry
from ess.sans.types import (
    BackgroundRun,
    DirectBeamFilename,
    Filename,
    PixelMaskFilename,
    SampleRun,
    TransmissionRun,
)

_registry = Registry(
    instrument='loki',
    files={
        # Files from LoKI@Larmor detector test experiment.
        # Original files are available at:
        # https://project.esss.dk/nextcloud/index.php/apps/files/?dir=/Data/LOKI_detector_test/nexus/2022-06-24_calibrated_nexus_files  # noqa: E501
        #
        # Background run 1 (no sample, sample holder/can only, no transmission monitor)
        '60248-2022-02-28_2215.nxs': 'md5:d9f17b95274a0fc6468df7e39df5bf03',
        # Sample run 1 (sample + sample holder/can, no transmission monitor in beam)
        '60250-2022-02-28_2215.nxs': 'md5:6a519ceaacbae702a6d08241e86799b1',
        # Sample run 2 (sample + sample holder/can, no transmission monitor in beam)
        '60339-2022-02-28_2215.nxs': 'md5:03c86f6389566326bb0cbbd80b8f8c4f',
        # Background transmission run (sample holder/can + transmission monitor)
        '60392-2022-02-28_2215.nxs': 'md5:9ecc1a9a2c05a880144afb299fc11042',
        # Background run 2 (no sample, sample holder/can only, no transmission monitor)
        '60393-2022-02-28_2215.nxs': 'md5:bf550d0ba29931f11b7450144f658652',
        # Sample transmission run (sample + sample holder/can + transmission monitor)
        '60394-2022-02-28_2215.nxs': 'md5:c40f38a62337d86957af925296c4c615',
        # ISIS polymer sample run
        "60395-2022-02-28_2215.nxs": "md5:d4c8ac05d0a015f2808089255e8ead3c",
        # AgBeh sample run
        "60387-2022-02-28_2215.nxs": "md5:157b937f00a5da133481b22b78ec7fa1",
        # AgBeh transmission run
        "60386-2022-02-28_2215.nxs": "md5:58ef33133e86e0026ee36f1a24deb464",
        # Porous silica sample run
        "60385-2022-02-28_2215.nxs": "md5:21275cb80c146d6c424bea81142b9a76",
        # Porous silica transmission run
        "60384-2022-02-28_2215.nxs": "md5:af04240efd3a245280f2d9f4846c6076",
        # deut-SDS sample run
        "60389-2022-02-28_2215.nxs": "md5:3126ec7a670ac603c6a5f4c756ddc5b7",
        # deut-SDS transmission run
        "60388-2022-02-28_2215.nxs": "md5:72737d0e796a5b7bb4241dd8157c5905",
        # Analytical model for the I(Q) of the Poly-Gauss sample
        'PolyGauss_I0-50_Rg-60.h5': 'md5:f5d60d9c2286cb197b8cd4dc82db3d7e',
        # XML file for the pixel mask
        'mask_new_July2022.xml': 'md5:421b6dc9db74126ffbc5d88164d017b0',
        # Direct beam from LoKI@Larmor detector test experiment
        'direct-beam-loki-all-pixels.h5': "md5:b85d7b486b312c5bb2a31d2bb6314f69",
        # Smaller files for unit tests
        'TEST_60248-2022-02-28_2215.nxs': 'md5:08201548d14dc0eb17afd0855111f1d2',
        'TEST_60250-2022-02-28_2215.nxs': 'md5:2b2ff9d68b98bfc76c57c016e6e5acce',
        'TEST_60339-2022-02-28_2215.nxs': 'md5:3c0199d05cddae03c7cd280a6c1d113d',
        'TEST_60384-2022-02-28_2215.nxs': 'md5:96adf8f02703b18879a73bb4b117c333',
        'TEST_60385-2022-02-28_2215.nxs': 'md5:c9c6af2ade8016f980dc1a6baabd7592',
        'TEST_60386-2022-02-28_2215.nxs': 'md5:100a517fc24b368505a72ab9cf3afa45',
        'TEST_60387-2022-02-28_2215.nxs': 'md5:2d81118553d98e0232da1855ba0159ab',
        'TEST_60388-2022-02-28_2215.nxs': 'md5:2339f75a9fa7b07b9a530dfcbf3e0d73',
        'TEST_60389-2022-02-28_2215.nxs': 'md5:bf28649903f34796ebd016309d752197',
        'TEST_60392-2022-02-28_2215.nxs': 'md5:2c4ac725f1a92f3d9ba404925416114b',
        'TEST_60393-2022-02-28_2215.nxs': 'md5:5ccaa9b47159144beb1440528147ab47',
        'TEST_60394-2022-02-28_2215.nxs': 'md5:d679a68ca2c86578548fec19ddb371d4',
        'TEST_60395-2022-02-28_2215.nxs': 'md5:179581a65951943f4c6bcf1943d32fee',
    },
    version='2',
)


def _get_path(filename: str, small: bool) -> str:
    prefix = 'TEST_' if small else ''
    return _registry.get_path(f'{prefix}{filename}')


def loki_tutorial_sample_run_60250(*, small: bool = False) -> Filename[SampleRun]:
    """Sample run with sample and sample holder/can, no transmission monitor in beam.

    Parameters
    ----------
    small:
        If True, return a smaller version of the file for unit tests.
        The file was created using the shrink_nexus_loki.py script in the repository
        top level `tools` folder.
    """
    return Filename[SampleRun](_get_path('60250-2022-02-28_2215.nxs', small=small))


def loki_tutorial_sample_run_60339(*, small: bool = False) -> Filename[SampleRun]:
    """Sample run with sample and sample holder/can, no transmission monitor in beam.

    Parameters
    ----------
    small:
        If True, return a smaller version of the file for unit tests.
        The file was created using the shrink_nexus_loki.py script in the repository
        top level `tools` folder."""
    return Filename[SampleRun](_get_path('60339-2022-02-28_2215.nxs', small=small))


def loki_tutorial_background_run_60248(
    *, small: bool = False
) -> Filename[BackgroundRun]:
    """Background run with sample holder/can only, no transmission monitor.

    Parameters
    ----------
    small:
        If True, return a smaller version of the file for unit tests.
        The file was created using the shrink_nexus_loki.py script in the repository
        top level `tools` folder.
    """
    return Filename[BackgroundRun](_get_path('60248-2022-02-28_2215.nxs', small=small))


def loki_tutorial_background_run_60393(
    *, small: bool = False
) -> Filename[BackgroundRun]:
    """Background run with sample holder/can only, no transmission monitor.

    Parameters
    ----------
    small:
        If True, return a smaller version of the file for unit tests.
        The file was created using the shrink_nexus_loki.py script in the repository
        top level `tools` folder.
    """
    return Filename[BackgroundRun](_get_path('60393-2022-02-28_2215.nxs', small=small))


def loki_tutorial_sample_transmission_run(
    *, small: bool = False
) -> Filename[TransmissionRun[SampleRun]]:
    """Sample transmission run (sample + sample holder/can + transmission monitor).

    Parameters
    ----------
    small:
        If True, return a smaller version of the file for unit tests.
        The file was created using the shrink_nexus_loki.py script in the repository
        top level `tools` folder.
    """
    return Filename[TransmissionRun[SampleRun]](
        _get_path('60394-2022-02-28_2215.nxs', small=small)
    )


def loki_tutorial_run_60392(
    *, small: bool = False
) -> Filename[TransmissionRun[BackgroundRun]]:
    """Background transmission run (sample holder/can + transmission monitor), also
    used as empty beam run.

    Parameters
    ----------
    small:
        If True, return a smaller version of the file for unit tests.
        The file was created using the shrink_nexus_loki.py script in the repository
        top level `tools` folder.
    """
    return Filename[TransmissionRun[BackgroundRun]](
        _get_path('60392-2022-02-28_2215.nxs', small=small)
    )


def loki_tutorial_isis_polymer_sample_run(
    *, small: bool = False
) -> Filename[SampleRun]:
    """Sample run with ISIS polymer sample.

    Parameters
    ----------
    small:
        If True, return a smaller version of the file for unit tests.
        The file was created using the shrink_nexus_loki.py script in the repository
        top level `tools` folder.
    """
    return Filename[SampleRun](_get_path("60395-2022-02-28_2215.nxs", small=small))


def loki_tutorial_isis_polymer_transmission_run(
    *, small: bool = False
) -> Filename[TransmissionRun[SampleRun]]:
    """Transmission run for ISIS polymer run.

    Parameters
    ----------
    small:
        If True, return a smaller version of the file for unit tests.
        The file was created using the shrink_nexus_loki.py script in the repository
        top level `tools` folder.
    """
    return Filename[TransmissionRun[SampleRun]](
        _get_path("60394-2022-02-28_2215.nxs", small=small)
    )


def loki_tutorial_agbeh_sample_run(*, small: bool = False) -> Filename[SampleRun]:
    """Sample run with AgBeh sample.

    Parameters
    ----------
    small:
        If True, return a smaller version of the file for unit tests.
        The file was created using the shrink_nexus_loki.py script in the repository
        top level `tools` folder.
    """
    return Filename[SampleRun](_get_path("60387-2022-02-28_2215.nxs", small=small))


def loki_tutorial_agbeh_transmission_run(
    *, small: bool = False
) -> Filename[TransmissionRun[SampleRun]]:
    """Transmission run for AgBeh run.

    Parameters
    ----------
    small:
        If True, return a smaller version of the file for unit tests.
        The file was created using the shrink_nexus_loki.py script in the repository
        top level `tools` folder.
    """
    return Filename[TransmissionRun[SampleRun]](
        _get_path("60386-2022-02-28_2215.nxs", small=small)
    )


def loki_tutorial_porous_silica_sample_run(
    *, small: bool = False
) -> Filename[SampleRun]:
    """Sample run with Porous silica sample.

    Parameters
    ----------
    small:
        If True, return a smaller version of the file for unit tests.
        The file was created using the shrink_nexus_loki.py script in the repository
        top level `tools` folder.
    """
    return Filename[SampleRun](_get_path("60385-2022-02-28_2215.nxs", small=small))


def loki_tutorial_porous_silica_transmission_run(
    *, small: bool = False
) -> Filename[TransmissionRun[SampleRun]]:
    """Transmission run for Porous silica run.

    Parameters
    ----------
    small:
        If True, return a smaller version of the file for unit tests.
        The file was created using the shrink_nexus_loki.py script in the repository
        top level `tools` folder.
    """
    return Filename[TransmissionRun[SampleRun]](
        _get_path("60384-2022-02-28_2215.nxs", small=small)
    )


def loki_tutorial_deut_sds_sample_run(*, small: bool = False) -> Filename[SampleRun]:
    """Sample run with deut-SDS sample.

    Parameters
    ----------
    small:
        If True, return a smaller version of the file for unit tests.
        The file was created using the shrink_nexus_loki.py script in the repository
        top level `tools` folder.
    """
    return Filename[SampleRun](_get_path("60389-2022-02-28_2215.nxs", small=small))


def loki_tutorial_deut_sds_transmission_run(
    *, small: bool = False
) -> Filename[TransmissionRun[SampleRun]]:
    """Transmission run for deut-SDS run.

    Parameters
    ----------
    small:
        If True, return a smaller version of the file for unit tests.
        The file was created using the shrink_nexus_loki.py script in the repository
        top level `tools` folder.
    """
    return Filename[TransmissionRun[SampleRun]](
        _get_path("60388-2022-02-28_2215.nxs", small=small)
    )


def loki_tutorial_mask_filenames() -> list[PixelMaskFilename]:
    """List of pixel mask filenames for the LoKI@Larmor detector test experiment."""
    return [
        PixelMaskFilename(_registry.get_path('mask_new_July2022.xml')),
    ]


def loki_tutorial_poly_gauss_I0() -> Path:
    """Analytical model for the I(Q) of the Poly-Gauss sample."""
    return Path(_registry.get_path('PolyGauss_I0-50_Rg-60.h5'))


def loki_tutorial_direct_beam_all_pixels() -> DirectBeamFilename:
    """File containing direct beam function computed using the direct beam iterations
    notebook, summing all pixels."""
    return DirectBeamFilename(_registry.get_path('direct-beam-loki-all-pixels.h5'))
