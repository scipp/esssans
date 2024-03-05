# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import numpy as np
import scipp as sc
from scipp.testing import assert_identical

from esssans.uncertainty import broadcast_with_upper_bound_variances


def test_broadcast_returns_original_if_no_new_dims():
    var = sc.ones(dims=['x', 'y'], shape=[2, 3], with_variances=True)
    assert (
        broadcast_with_upper_bound_variances(var, template=sc.empty(sizes={'x': 2}))
        is var
    )
    assert (
        broadcast_with_upper_bound_variances(var, template=sc.empty(sizes={'y': 3}))
        is var
    )
    assert (
        broadcast_with_upper_bound_variances(
            var, template=sc.empty(sizes={'y': 3, 'x': 2})
        )
        is var
    )
    assert (
        broadcast_with_upper_bound_variances(
            var, template=sc.empty(sizes={'x': 2, 'y': 3})
        )
        is var
    )


def test_broadcast_returns_original_if_no_variances():
    var = sc.ones(dims=['x'], shape=[2], with_variances=False)
    assert (
        broadcast_with_upper_bound_variances(var, template=sc.empty(sizes={'y': 3}))
        is var
    )


def test_broadcast_scales_variances_by_new_subspace_volume():
    x = sc.linspace('x', 0.0, 1.0, 2)
    y = sc.linspace('y', 0.0, 2.0, 3)
    values = x * y
    var = values.copy()
    var.variances = var.values
    expected = sc.ones(dims=['z'], shape=[1]) * values
    expected.variances = 1 * expected.values
    assert_identical(
        broadcast_with_upper_bound_variances(var, template=sc.empty(sizes={'z': 1})),
        expected,
    )
    expected = sc.ones(dims=['z'], shape=[2]) * values
    expected.variances = 2 * expected.values
    assert_identical(
        broadcast_with_upper_bound_variances(var, template=sc.empty(sizes={'z': 2})),
        expected,
    )
    expected = sc.ones(dims=['z'], shape=[2]) * values
    expected.variances = 2 * expected.values
    assert_identical(
        broadcast_with_upper_bound_variances(
            var, template=sc.empty(sizes={'y': 3, 'z': 2})
        ),
        expected.transpose(['y', 'z', 'x']),
    )


def test_broadcast_new_subspace_volume_with_masks():
    # Data: A A A A A  Template: X
    #                            X
    #                            B
    #                            B
    #                            B
    #                            B
    x = np.linspace(0.0, 1.0, 5)
    a = sc.array(dims=['x'], values=x, variances=x)
    b = sc.DataArray(
        data=sc.ones(sizes={'y': 6}),
        masks={
            'm': sc.array(dims=['y'], values=[True, True, False, False, False, False])
        },
    )
    expected = sc.values(b.data) * sc.values(a)
    expected.variances = (
        np.broadcast_to(x, expected.shape) * 4
    )  # 4 non-masked elements in b
    assert_identical(broadcast_with_upper_bound_variances(a, template=b), expected)


def test_broadcast_with_masks_and_a_common_dimension():
    # Data: A A A A A  Template: B B B B B
    #                            X X X X X
    #                            B B B B B
    #                            B B B B B
    #                            X X X X X
    #                            B B B B B
    x = np.linspace(0.0, 1.0, 5)
    a = sc.array(dims=['x'], values=x, variances=x)
    b = sc.DataArray(
        data=sc.ones(sizes={'y': 6, 'x': 5}),
        masks={
            'm': sc.array(dims=['y'], values=[False, True, False, False, True, False])
        },
    )
    expected = sc.values(b.data) * sc.values(a)
    expected.variances = np.broadcast_to(x, expected.shape) * 4
    assert_identical(broadcast_with_upper_bound_variances(a, template=b), expected)


def test_broadcast_with_masks_and_two_common_dimensions():
    x = np.linspace(0.0, 5.0, 10).reshape(5, 2)
    a = sc.array(dims=['x', 'z'], values=x, variances=x)
    b = sc.DataArray(
        data=sc.ones(sizes={'y': 6, 'x': 5, 'z': 2}),
        masks={
            'm': sc.array(dims=['y'], values=[False, True, True, False, True, False])
        },
    )
    expected = sc.values(b.data) * sc.values(a)
    expected.variances = np.broadcast_to(x, expected.shape) * 3
    assert_identical(broadcast_with_upper_bound_variances(a, template=b), expected)


def test_broadcast_with_2d_mask():
    # Data: A A A A A  Template: B X X B B
    #                            B B B B B
    #                            X X X X X
    #                            X X X X X
    #                            B B B B B
    #                            B B B B X

    # TODO: It is not so clear what to do here. This would be, for example, a case where
    # the direct beam depends on wavelength and layer, while the solid angle depends on
    # layer and straw, and the mask on the solid angle depends on both layer and straw.
    # Do we just count the number of non-masked elements in the template?

    x = np.linspace(0.0, 60.0, 5)
    a = sc.array(dims=['x'], values=x, variances=x)
    b = sc.DataArray(
        data=sc.ones(sizes={'y': 6, 'x': 5}),
        masks={
            'm': sc.array(
                dims=['y', 'x'],
                values=[
                    [False, True, True, False, False],
                    [False, False, False, False, False],
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                    [False, False, False, False, False],
                    [False, False, False, False, True],
                ],
            )
        },
    )
    expected = sc.values(b.data) * sc.values(a)
    expected.variances = np.broadcast_to(x, expected.shape) * 17 / 5
    assert_identical(broadcast_with_upper_bound_variances(a, template=b), expected)


def test_broadcast_with_2d_mask_with_extra_dimension_on_input():
    # Data: A A A A A  Template: B X X B B
    #      A A A A A             B B B B B
    #                            X X X X X
    #                            X X X X X
    #                            B B B B B
    #                            B B B B X

    x = np.linspace(0.0, 60.0, 10).reshape(5, 2)
    a = sc.array(dims=['x', 'z'], values=x, variances=x)
    b = sc.DataArray(
        data=sc.ones(sizes={'y': 6, 'x': 5}),
        masks={
            'm': sc.array(
                dims=['y', 'x'],
                values=[
                    [False, True, True, False, False],
                    [False, False, False, False, False],
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                    [False, False, False, False, False],
                    [False, False, False, False, True],
                ],
            )
        },
    )
    expected = sc.values(b.data) * sc.values(a)
    # Only divide by dimension sizes of a that are also found in b
    expected.variances = np.broadcast_to(x, expected.shape) * 17 / 5
    assert_identical(broadcast_with_upper_bound_variances(a, template=b), expected)
