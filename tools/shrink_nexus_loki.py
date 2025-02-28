# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import h5py as h5
import numpy as np

files = [
    '60248-2022-02-28_2215.nxs',
    '60250-2022-02-28_2215.nxs',
    '60339-2022-02-28_2215.nxs',
    '60384-2022-02-28_2215.nxs',
    '60385-2022-02-28_2215.nxs',
    '60386-2022-02-28_2215.nxs',
    '60387-2022-02-28_2215.nxs',
    '60388-2022-02-28_2215.nxs',
    '60389-2022-02-28_2215.nxs',
    '60392-2022-02-28_2215.nxs',
    '60393-2022-02-28_2215.nxs',
    '60394-2022-02-28_2215.nxs',
    '60395-2022-02-28_2215.nxs',
]

DETECTOR_BANK_SIZES = {
    'larmor_detector': {'layer': 4, 'tube': 32, 'straw': 7, 'pixel': 512}
}

keys = DETECTOR_BANK_SIZES.keys()

for fname in files:
    print(fname)  # noqa: T201

    with h5.File(fname, 'r+') as ds:
        for key in keys:
            print(key)  # noqa: T201
            base_path = f"entry/instrument/{key}"
            det_nums = ds[base_path + '/detector_number'][()]
            folded = det_nums.reshape(list(DETECTOR_BANK_SIZES[key].values()))
            # First and last layer
            # Every other tube
            # First, middle and last straw
            # Every fourth pixel
            keep = folded[::3, ::2, ::3, ::4]

            tmp = base_path + "/tmp"  # noqa: S108
            sel = np.isin(det_nums, keep)
            for field in (
                'detector_number',
                'x_pixel_offset',
                'y_pixel_offset',
                'z_pixel_offset',
            ):
                here = base_path + f"/{field}"
                old = ds[here][()]
                ds[tmp] = ds[here]
                del ds[here]  # delete old, differently sized dataset
                ds.create_dataset(here, data=old[sel])
                ds[here].attrs.update(ds[tmp].attrs)
                del ds[tmp]

            event_path = base_path + f"/{key}_events"
            tmp = event_path + "/temp_path"

            id_path = event_path + "/event_id"
            evids = ds[id_path][()]
            sel = np.isin(evids, keep)

            ds[tmp] = ds[id_path]
            del ds[id_path]
            ds.create_dataset(id_path, data=evids[sel])
            ds[id_path].attrs.update(ds[tmp].attrs)
            del ds[tmp]

            eto_path = event_path + "/event_time_offset"
            etos = ds[eto_path][()]
            ds[tmp] = ds[eto_path]
            del ds[eto_path]
            ds.create_dataset(eto_path, data=etos[sel])
            ds[eto_path].attrs.update(ds[tmp].attrs)
            del ds[tmp]

            # Rebuild event_index
            index_path = event_path + "/event_index"
            eind = ds[index_path][()]
            etz = ds[event_path + '/event_time_zero'][()]
            broadcasted = np.repeat(
                etz, np.concat([np.diff(eind), [len(etos) - eind[-1]]])
            )
            filt = broadcasted[sel]
            change_indices = np.where(np.diff(filt) != 0)[0] + 1
            # Compute the run lengths
            repeats = np.diff(np.append([0], np.append(change_indices, filt.size)))
            ds[tmp] = ds[index_path]
            del ds[index_path]
            new_index = np.cumsum(np.concat([[0], repeats]))[:-1]
            # It could be that now, some pulses have no events, but we need to make sure
            # event_index has the same size as event_time_zero
            if len(new_index) < len(etz):
                new_index = np.concatenate(
                    [new_index, np.repeat(new_index[-1], len(etz) - len(new_index))]
                )
            ds.create_dataset(index_path, data=new_index)
            ds[index_path].attrs.update(ds[tmp].attrs)
            del ds[tmp]

        instr_path = "entry/instrument"
        # Monitors
        for m in range(2):
            mon1_path = instr_path + f'/monitor_{m + 1}/monitor_{m + 1}_events'
            # Select 10 times less pulses
            ev_ind_path = mon1_path + '/event_index'
            ev_ind = ds[ev_ind_path][()]
            npulses = max(1, len(ev_ind) // 10)
            n = ev_ind[npulses]
            tmp = mon1_path + "/temp_path"
            # event_index
            ds[tmp] = ds[ev_ind_path]
            del ds[ev_ind_path]
            ds.create_dataset(ev_ind_path, data=ev_ind[0:npulses])
            ds[ev_ind_path].attrs.update(ds[tmp].attrs)
            del ds[tmp]
            # event_id
            ev_id_path = mon1_path + '/event_id'
            ev_ids = ds[ev_id_path][()]
            ds[tmp] = ds[ev_id_path]
            del ds[ev_id_path]
            ds.create_dataset(ev_id_path, data=ev_ids[:n])
            ds[ev_id_path].attrs.update(ds[tmp].attrs)
            del ds[tmp]
            # event_time_offset
            eto_path = mon1_path + '/event_time_offset'
            etos = ds[eto_path][()]
            ds[tmp] = ds[eto_path]
            del ds[eto_path]
            ds.create_dataset(eto_path, data=etos[:n])
            ds[eto_path].attrs.update(ds[tmp].attrs)
            del ds[tmp]
            # event_time_zero
            etz_path = mon1_path + '/event_time_zero'
            etz = ds[etz_path][()]
            ds[tmp] = ds[etz_path]
            del ds[etz_path]
            ds.create_dataset(etz_path, data=etos[0:npulses])
            ds[etz_path].attrs.update(ds[tmp].attrs)
            del ds[tmp]
