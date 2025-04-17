# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os.path as op
from os import PathLike
from pathlib import Path

import numpy as np

from mne.utils.check import _check_option

from mne._fiff._digitization import _ensure_fiducials_head
from mne._fiff.constants import FIFF
from mne._fiff.meas_info import create_info
from mne._fiff.pick import _PICK_TYPES_KEYS
from mne._fiff.utils import _find_channels
from mne.annotations import read_annotations
from mne.channels import make_dig_montage
from mne.defaults import DEFAULTS
from mne.utils import (
    Bunch,
    _check_fname,
    _check_head_radius,
    fill_doc,
    logger,
    verbose,
    warn,
)
from mne.io.base import BaseRaw
from mne.utils import Bunch, _import_pymatreader_funcs
from scipy.io import loadmat
from scipy.io.matlab import MatlabOpaque, MatlabFunction

# just fix the scaling for now, EEGLAB doesn't seem to provide this info
CAL = 1e-6


def _check_eeglab_fname(fname, dataname):
    """Check whether the filename is valid.

    Check if the file extension is ``.fdt`` (older ``.dat`` being invalid) or
    whether the ``EEG.data`` filename exists. If ``EEG.data`` file is absent
    the set file name with .set changed to .fdt is checked.
    """
    fmt = str(op.splitext(dataname)[-1])
    if fmt == ".dat":
        raise NotImplementedError(
            "Old data format .dat detected. Please update your EEGLAB "
            "version and resave the data in .fdt format"
        )

    basedir = op.dirname(fname)
    data_fname = op.join(basedir, dataname)
    if not op.exists(data_fname):
        fdt_from_set_fname = op.splitext(fname)[0] + ".fdt"
        if op.exists(fdt_from_set_fname):
            data_fname = fdt_from_set_fname
            msg = (
                "Data file name in EEG.data ({}) is incorrect, the file "
                "name must have changed on disk, using the correct file "
                "name ({})."
            )
            warn(msg.format(dataname, op.basename(fdt_from_set_fname)))
        elif not data_fname == fdt_from_set_fname:
            msg = "Could not find the .fdt data file, tried {} and {}."
            raise FileNotFoundError(msg.format(data_fname, fdt_from_set_fname))
    return data_fname


def _check_load_mat(fname, uint16_codec):
    """Check if the mat struct contains 'EEG'."""
    eeg = _readmat(fname, uint16_codec=uint16_codec)
    if "ALLEEG" in eeg:
        raise NotImplementedError(
            "Loading an ALLEEG array is not supported. Please contact"
            "mne-python developers for more information."
        )
    if "EEG" in eeg:  # fields are contained in EEG structure
        eeg = eeg["EEG"]
    eeg = eeg.get("EEG", eeg)  # handle nested EEG structure
    eeg = Bunch(**eeg)
    eeg.trials = int(eeg.trials)
    eeg.nbchan = int(eeg.nbchan)
    eeg.pnts = int(eeg.pnts)
    return eeg


def _to_loc(ll):
    """Check if location exists."""
    if isinstance(ll, (int, float)) or len(ll) > 0:
        return ll
    else:
        return np.nan


def _eeg_has_montage_information(eeg):
    try:
        from scipy.io.matlab import mat_struct
    except ImportError:  # SciPy < 1.8
        from scipy.io.matlab.mio5_params import mat_struct
    if not len(eeg.chanlocs):
        has_pos = False
    else:
        pos_fields = ["X", "Y", "Z"]
        if isinstance(eeg.chanlocs[0], mat_struct):
            has_pos = all(hasattr(eeg.chanlocs[0], fld) for fld in pos_fields)
        elif isinstance(eeg.chanlocs[0], np.ndarray):
            # Old files
            has_pos = all(fld in eeg.chanlocs[0].dtype.names for fld in pos_fields)
        elif isinstance(eeg.chanlocs[0], dict):
            # new files
            has_pos = all(fld in eeg.chanlocs[0] for fld in pos_fields)
        else:
            has_pos = False  # unknown (sometimes we get [0, 0])

    return has_pos


def _get_montage_information(eeg, get_pos, *, montage_units):
    """Get channel name, type and montage information from ['chanlocs']."""
    ch_names, ch_types, pos_ch_names, pos = list(), list(), list(), list()
    unknown_types = dict()
    for chanloc in eeg.chanlocs:
        # channel name
        ch_names.append(chanloc["labels"])

        # channel type
        ch_type = "eeg"
        try_type = chanloc.get("type", None)
        if isinstance(try_type, str):
            try_type = try_type.strip().lower()
            if try_type in _PICK_TYPES_KEYS:
                ch_type = try_type
            else:
                if try_type in unknown_types:
                    unknown_types[try_type].append(chanloc["labels"])
                else:
                    unknown_types[try_type] = [chanloc["labels"]]
        ch_types.append(ch_type)

        # channel loc
        if get_pos:
            loc_x = _to_loc(chanloc["X"])
            loc_y = _to_loc(chanloc["Y"])
            loc_z = _to_loc(chanloc["Z"])
            locs = np.r_[-loc_y, loc_x, loc_z]
            pos_ch_names.append(chanloc["labels"])
            pos.append(locs)

    # warn if unknown types were provided
    if len(unknown_types):
        warn(
            "Unknown types found, setting as type EEG:\n"
            + "\n".join(
                [
                    f"{key}: {sorted(unknown_types[key])}"
                    for key in sorted(unknown_types)
                ]
            )
        )

    lpa, rpa, nasion = None, None, None
    if hasattr(eeg, "chaninfo") and isinstance(eeg.chaninfo["nodatchans"], dict):
        nodatchans = eeg.chaninfo["nodatchans"]
        types = nodatchans.get("type", [])
        descriptions = nodatchans.get("description", [])
        xs = nodatchans.get("X", [])
        ys = nodatchans.get("Y", [])
        zs = nodatchans.get("Z", [])

        for type_, description, x, y, z in zip(types, descriptions, xs, ys, zs):
            if type_ != "FID":
                continue
            if description == "Nasion":
                nasion = np.array([x, y, z])
            elif description == "Right periauricular point":
                rpa = np.array([x, y, z])
            elif description == "Left periauricular point":
                lpa = np.array([x, y, z])

    # Always check this even if it's not used
    _check_option("montage_units", montage_units, ("m", "dm", "cm", "mm", "auto"))
    if pos_ch_names:
        pos_array = np.array(pos, float)
        pos_array.shape = (-1, 3)

        # roughly estimate head radius and check if its reasonable
        is_nan_pos = np.isnan(pos).any(axis=1)
        if not is_nan_pos.all():
            mean_radius = np.mean(np.linalg.norm(pos_array[~is_nan_pos], axis=1))
            scale_units = _handle_montage_units(montage_units, mean_radius)
            mean_radius *= scale_units
            pos_array *= scale_units
            additional_info = (
                " Check if the montage_units argument is correct (the default "
                'is "mm", but your channel positions may be in different units'
                ")."
            )
            _check_head_radius(mean_radius, add_info=additional_info)

        montage = make_dig_montage(
            ch_pos=dict(zip(ch_names, pos_array)),
            coord_frame="head",
            lpa=lpa,
            rpa=rpa,
            nasion=nasion,
        )
        _ensure_fiducials_head(montage.dig)
    else:
        montage = None

    return ch_names, ch_types, montage


def _get_info(eeg, *, eog, montage_units):
    """Get measurement info."""
    # add the ch_names and info['chs'][idx]['loc']
    if not isinstance(eeg.chanlocs, np.ndarray) and eeg.nbchan == 1:
        eeg.chanlocs = [eeg.chanlocs]

    if isinstance(eeg.chanlocs, dict):
        eeg.chanlocs = _dol_to_lod(eeg.chanlocs)

    eeg_has_ch_names_info = len(eeg.chanlocs) > 0

    if eeg_has_ch_names_info:
        has_pos = _eeg_has_montage_information(eeg)
        ch_names, ch_types, eeg_montage = _get_montage_information(
            eeg, has_pos, montage_units=montage_units
        )
        update_ch_names = False
    else:  # if eeg.chanlocs is empty, we still need default chan names
        ch_names = [f"EEG {ii:03d}" for ii in range(eeg.nbchan)]
        ch_types = "eeg"
        eeg_montage = None
        update_ch_names = True

    info = create_info(ch_names, sfreq=eeg.srate, ch_types=ch_types)

    eog = _find_channels(ch_names, ch_type="EOG") if eog == "auto" else eog
    for idx, ch in enumerate(info["chs"]):
        ch["cal"] = CAL
        if ch["ch_name"] in eog or idx in eog:
            ch["coil_type"] = FIFF.FIFFV_COIL_NONE
            ch["kind"] = FIFF.FIFFV_EOG_CH

    return info, eeg_montage, update_ch_names


def _set_dig_montage_in_init(self, montage):
    """Set EEG sensor configuration and head digitization from when init.

    This is done from the information within fname when
    read_raw_eeglab(fname) or read_epochs_eeglab(fname).
    """
    if montage is None:
        self.set_montage(None)
    else:
        missing_channels = set(self.ch_names) - set(montage.ch_names)
        ch_pos = dict(
            zip(list(missing_channels), np.full((len(missing_channels), 3), np.nan))
        )
        self.set_montage(montage + make_dig_montage(ch_pos=ch_pos, coord_frame="head"))


def _handle_montage_units(montage_units, mean_radius):
    if montage_units == "auto":
        # radius should be between 0.05 and 0.11 meters
        if mean_radius < 0.25:
            montage_units = "m"
        elif mean_radius < 2.5:
            montage_units = "dm"
        elif mean_radius < 25:
            montage_units = "cm"
        else:  # mean_radius >= 25
            montage_units = "mm"
    prefix = montage_units[:-1]
    scale_units = 1 / DEFAULTS["prefixes"][prefix]
    return scale_units


@fill_doc
def read_raw_eeglab(
    input_fname,
    eog=(),
    preload=False,
    uint16_codec=None,
    montage_units="auto",
    verbose=None,
) -> "RawEEGLAB":
    r"""Read an EEGLAB .set file.

    Parameters
    ----------
    input_fname : path-like
        Path to the ``.set`` file. If the data is stored in a separate ``.fdt``
        file, it is expected to be in the same folder as the ``.set`` file.
    eog : list | tuple | ``'auto'``
        Names or indices of channels that should be designated EOG channels.
        If 'auto', the channel names containing ``EOG`` or ``EYE`` are used.
        Defaults to empty tuple.
    %(preload)s
        Note that ``preload=False`` will be effective only if the data is
        stored in a separate binary file.
    %(uint16_codec)s
    %(montage_units)s

        .. versionchanged:: 1.6
           Support for ``'auto'`` was added and is the new default.
    %(verbose)s

    Returns
    -------
    raw : instance of RawEEGLAB
        A Raw object containing EEGLAB .set data.
        See :class:`mne.io.Raw` for documentation of attributes and methods.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods of RawEEGLAB.

    Notes
    -----
    .. versionadded:: 0.11.0
    """
    return RawEEGLAB(
        input_fname=input_fname,
        preload=preload,
        eog=eog,
        uint16_codec=uint16_codec,
        montage_units=montage_units,
        verbose=verbose,
    )


@fill_doc
class RawEEGLAB(BaseRaw):
    r"""Raw object from EEGLAB .set file.

    Parameters
    ----------
    input_fname : path-like
        Path to the ``.set`` file. If the data is stored in a separate ``.fdt``
        file, it is expected to be in the same folder as the ``.set`` file.
    eog : list | tuple | 'auto'
        Names or indices of channels that should be designated EOG channels.
        If 'auto', the channel names containing ``EOG`` or ``EYE`` are used.
        Defaults to empty tuple.
    %(preload)s
        Note that preload=False will be effective only if the data is stored
        in a separate binary file.
    %(uint16_codec)s
    %(montage_units)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods.

    Notes
    -----
    .. versionadded:: 0.11.0
    """

    @verbose
    def __init__(
        self,
        input_fname,
        eog=(),
        preload=False,
        *,
        uint16_codec=None,
        montage_units="auto",
        verbose=None,
    ):
        input_fname = str(_check_fname(input_fname, "read", True, "input_fname"))
        eeg = _check_load_mat(input_fname, uint16_codec)

        last_samps = [eeg.pnts - 1]
        info, eeg_montage, _ = _get_info(eeg, eog=eog, montage_units=montage_units)

        # read the data
        if isinstance(eeg.data, str):
            data_fname = _check_eeglab_fname(input_fname, eeg.data)
            logger.info(f"Reading {data_fname}")

            super().__init__(
                info,
                preload,
                filenames=[data_fname],
                last_samps=last_samps,
                orig_format="double",
                verbose=verbose,
            )
        else:
            if preload is False or isinstance(preload, str):
                warn(
                    "Data will be preloaded. preload=False or a string "
                    "preload is not supported when the data is stored in "
                    "the .set file"
                )
            # can't be done in standard way with preload=True because of
            # different reading path (.set file)
            if eeg.nbchan == 1 and len(eeg.data.shape) == 1:
                n_chan, n_times = [1, eeg.data.shape[0]]
            else:
                n_chan, n_times = eeg.data.shape
            data = np.empty((n_chan, n_times), dtype=float)
            data[:n_chan] = eeg.data
            data *= CAL
            super().__init__(
                info,
                data,
                filenames=[input_fname],
                last_samps=last_samps,
                orig_format="double",
                verbose=verbose,
            )

        # create event_ch from annotations
        # annot = read_annotations(input_fname, uint16_codec=uint16_codec)
        # self.set_annotations(annot)
        # _check_boundary(annot, None)

        _set_dig_montage_in_init(self, eeg_montage)

        # latencies = np.round(annot.onset * self.info["sfreq"])
        # _check_latencies(latencies)


def _check_boundary(annot, event_id):
    if event_id is None:
        event_id = dict()
    if "boundary" in annot.description and "boundary" not in event_id:
        warn(
            "The data contains 'boundary' events, indicating data "
            "discontinuities. Be cautious of filtering and epoching around "
            "these events."
        )


def _check_latencies(latencies):
    if (latencies < -1).any():
        raise ValueError(
            "At least one event sample index is negative. Please"
            " check if EEG.event.sample values are correct."
        )
    if (latencies == -1).any():
        warn(
            "At least one event has a sample index of -1. This usually is "
            "a consequence of how eeglab handles event latency after "
            "resampling - especially when you had a boundary event at the "
            "beginning of the file. Please make sure that the events at "
            "the very beginning of your EEGLAB file can be safely dropped "
            "(e.g., because they are boundary events)."
        )


def _dol_to_lod(dol):
    """Convert a dict of lists to a list of dicts."""
    return [
        {key: dol[key][ii] for key in dol.keys()}
        for ii in range(len(dol[list(dol.keys())[0]]))
    ]


def _todict_from_np_struct(data):  # taken from pymatreader.utils
    data_dict = {}

    for cur_field_name in data.dtype.names:
        try:
            n_items = len(data[cur_field_name])
            cur_list = []

            for idx in np.arange(n_items):
                cur_value = data[cur_field_name].item(idx)
                cur_value = _check_for_scipy_mat_struct(cur_value)
                cur_list.append(cur_value)

            data_dict[cur_field_name] = cur_list
        except TypeError:
            cur_value = data[cur_field_name].item(0)
            cur_value = _check_for_scipy_mat_struct(cur_value)
            data_dict[cur_field_name] = cur_value

    return data_dict


def _handle_scipy_ndarray(data):  # taken from pymatreader.utils
    if data.dtype == np.dtype("object") and not isinstance(data, MatlabFunction):
        as_list = []
        for element in data:
            as_list.append(_check_for_scipy_mat_struct(element))
        data = as_list
    elif isinstance(data.dtype.names, tuple):
        data = _todict_from_np_struct(data)
        data = _check_for_scipy_mat_struct(data)

    if isinstance(data, np.ndarray):
        data = np.array(data)

    return data


def _check_for_scipy_mat_struct(data):  # taken from pymatreader.utils
    """Convert all scipy.io.matlab.mio5_params.mat_struct elements."""
    if isinstance(data, dict):
        for key in data:
            data[key] = _check_for_scipy_mat_struct(data[key])

    if isinstance(data, MatlabOpaque):
        try:
            if data[0][2] == b"string":
                return None
        except IndexError:
            pass

    if isinstance(data, np.ndarray):
        data = _handle_scipy_ndarray(data)

    return data


def _readmat(fname, uint16_codec=None):
    try:
        read_mat = _import_pymatreader_funcs("EEGLAB I/O")
    except RuntimeError:  # pymatreader not installed
        eeg = loadmat(fname, squeeze_me=True, mat_dtype=False)
        return _check_for_scipy_mat_struct(eeg)
    else:
        return read_mat(fname, uint16_codec=uint16_codec)


def _check_load_mat(fname, uint16_codec=None):
    """Check if the mat struct contains 'EEG'."""
    eeg = _readmat(fname, uint16_codec=uint16_codec)
    if "ALLEEG" in eeg:
        raise NotImplementedError(
            "Loading an ALLEEG array is not supported. Please contact"
            "mne-python developers for more information."
        )
    if "EEG" in eeg:  # fields are contained in EEG structure
        eeg = eeg["EEG"]
    eeg = eeg.get("EEG", eeg)  # handle nested EEG structure
    eeg = Bunch(**eeg)
    eeg.trials = int(eeg.trials)
    eeg.nbchan = int(eeg.nbchan)
    eeg.pnts = int(eeg.pnts)
    return eeg
