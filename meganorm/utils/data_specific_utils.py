import mne
import numpy as np
import json
import os
import glob
from pathlib import Path
import numpy as np
import nibabel as nib
import mne
from mne.io.constants import FIFF


def _ast_get_rs_events(raw, start_channel="STI001", end_channel="STI002"):
    """Returns paired (start_sample, end_sample) for each resting block."""
    starts = mne.find_events(raw, stim_channel=start_channel, shortest_event=1)[:, 0]
    ends = mne.find_events(raw, stim_channel=end_channel, shortest_event=1)[:, 0]

    assert len(starts) == len(ends), f"{len(starts)} starts vs {len(ends)} ends"

    return list(zip(starts, ends))


def _ast_get_rs_block(raw, block_index, start_channel="STI001", end_channel="STI002"):
    """block_index: 0 = first block, 1 = second block."""
    blocks = _ast_get_rs_events(raw, start_channel, end_channel)
    start_samp, end_samp = blocks[block_index]

    tmin = (start_samp - raw.first_samp) / raw.info["sfreq"]
    tmax = (end_samp - raw.first_samp) / raw.info["sfreq"]

    return raw.copy().crop(tmin=tmin, tmax=tmax)


def _trans_from_nimh(raw, coordsystem_json, subject, subjects_dir):

    cs = json.load(open(coordsystem_json))
    alm = cs["AnatomicalLandmarkCoordinates"]

    # The MRI landmarks are in scanner space, but MNE works in FreeSurfer surface RAS space; a different origin
    t1 = nib.load(f"{subjects_dir}/{subject}/mri/T1.mgz")
    scanner_to_surf = t1.header.get_vox2ras_tkr() @ np.linalg.inv(
        t1.header.get_vox2ras()
    )

    def lps_mm_to_surf_m(p):
        p_ras = np.array(p, float) * [-1, -1, 1]
        return (scanner_to_surf @ np.r_[p_ras, 1.0])[:3] / 1000.0

    fids = [
        dict(
            kind=FIFF.FIFFV_POINT_CARDINAL,
            ident=FIFF.FIFFV_POINT_LPA,
            r=lps_mm_to_surf_m(alm["LPA"]),
            coord_frame=FIFF.FIFFV_COORD_MRI,
        ),
        dict(
            kind=FIFF.FIFFV_POINT_CARDINAL,
            ident=FIFF.FIFFV_POINT_NASION,
            r=lps_mm_to_surf_m(alm["NAS"]),
            coord_frame=FIFF.FIFFV_COORD_MRI,
        ),
        dict(
            kind=FIFF.FIFFV_POINT_CARDINAL,
            ident=FIFF.FIFFV_POINT_RPA,
            r=lps_mm_to_surf_m(alm["RPA"]),
            coord_frame=FIFF.FIFFV_COORD_MRI,
        ),
    ]

    coreg = mne.coreg.Coregistration(
        raw.info, subject=subject, subjects_dir=subjects_dir, fiducials=fids
    )  # list, not a filename
    coreg.fit_fiducials()
    return coreg, fids



def separate_eyes_open_close_eeglab(
    input_base_path,
    output_base_path,
    annotation_description_open,
    annotation_description_close,
    trim_before=5,
    trim_after=5,
):
    """
    Split resting-state EEGLAB recordings into separate eyes-open and
    eyes-closed files based on annotations.

    Scans `input_base_path` for BIDS-style resting-state ``.set`` files,
    extracts and trims annotated eyes-open and eyes-closed segments,
    concatenates each condition's segments, and writes them out as new
    EEGLAB ``.set`` files under a subject-specific folder in
    `output_base_path`.

    Parameters
    ----------
    input_base_path : str
        Root directory containing subject subfolders with resting-state
        EEGLAB recordings, matched via the pattern
        ``*/eeg/*_task-rest_eeg.set``.
    output_base_path : str
        Root directory where the separated eyes-open and eyes-closed
        files will be saved, created if it does not already exist.
    annotation_description_open : str
        Annotation description label marking eyes-open segments.
    annotation_description_close : str
        Annotation description label marking eyes-closed segments.
    trim_before : float, optional
        Duration in seconds to trim from the start of each annotated
        segment. Default is 5.
    trim_after : float, optional
        Duration in seconds to trim from the end of each annotated
        segment. Default is 5.

    Returns
    -------
    None
    """
    # Ensure output directory exists
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    search_pattern = os.path.join(input_base_path, "*/eeg/*_task-rest_eeg.set")
    raw_set_paths = glob.glob(
        search_pattern, recursive=True
    )  # Use glob to find all .set files in the input directory

    # Loop through all found .set files
    for set_path in raw_set_paths:
        subject_id = Path(set_path).parts[
            -3
        ]  # Extract subject number from the file path
        subject_output_path = os.path.join(
            output_base_path, subject_id, "eeg"
        )  # Create the subject-specific output path

        # Ensure output directory for the subject exists
        if not os.path.exists(subject_output_path):
            os.makedirs(subject_output_path)

        # Load the raw .set file (EEGLAB format)
        raw = mne.io.read_raw(set_path, preload=True)

        # Extract annotations
        annotations = raw.annotations

        # Separate eyes open and eyes closed events
        eyes_open_events = annotations[
            annotations.description == annotation_description_open
        ]
        eyes_closed_events = annotations[
            annotations.description == annotation_description_close
        ]

        # Extract and concatenate eyes open segments
        eyes_open_data = []
        for onset, duration in zip(eyes_open_events.onset, eyes_open_events.duration):

            if duration <= trim_before + trim_after:
                print(
                    f"Skipping event with onset {onset} and duration {duration} (invalid after trimming)"
                )
                continue

            # Trim the first 5s and last 5s from each event
            trimmed_onset = onset + trim_before
            trimmed_duration = duration - trim_before - trim_after
            start_sample = int(trimmed_onset * raw.info["sfreq"])
            stop_sample = int((trimmed_onset + trimmed_duration) * raw.info["sfreq"])
            eyes_open_data.append(raw[:, start_sample:stop_sample][0])

        if eyes_open_data:
            eyes_open_data_concat = np.concatenate(eyes_open_data, axis=1)
            raw_eyes_open = mne.io.RawArray(eyes_open_data_concat, raw.info)

            # Save eyes open data as a new .set file
            eyes_open_file_path = os.path.join(
                subject_output_path, f"{subject_id}_task-eyesopen_eeg.set"
            )
            mne.export.export_raw(
                eyes_open_file_path, raw_eyes_open, fmt="eeglab", overwrite=True
            )

        # Extract and concatenate eyes closed segments
        eyes_closed_data = []
        for onset, duration in zip(
            eyes_closed_events.onset, eyes_closed_events.duration
        ):

            if duration <= trim_before + trim_after:
                print(
                    f"Skipping event with onset {onset} and duration {duration} (invalid after trimming)"
                )
                continue

            trimmed_onset = onset + trim_before
            trimmed_duration = duration - trim_before - trim_after
            start_sample = int(trimmed_onset * raw.info["sfreq"])
            stop_sample = int((trimmed_onset + trimmed_duration) * raw.info["sfreq"])
            eyes_closed_data.append(raw[:, start_sample:stop_sample][0])

        if eyes_closed_data:
            eyes_closed_data_concat = np.concatenate(eyes_closed_data, axis=1)
            raw_eyes_closed = mne.io.RawArray(eyes_closed_data_concat, raw.info)

            # Save eyes closed data as a new .set file
            eyes_closed_file_path = os.path.join(
                subject_output_path, f"{subject_id}_task-eyesclosed_eeg.set"
            )
            mne.export.export_raw(
                eyes_closed_file_path, raw_eyes_closed, fmt="eeglab", overwrite=True
            )