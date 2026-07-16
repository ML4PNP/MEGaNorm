import mne
import numpy as np
import json
import numpy as np
import nibabel as nib
import mne
from mne.io.constants import FIFF

def _ast_get_rs_events(raw, start_channel='STI001', end_channel='STI002'):
    """Returns paired (start_sample, end_sample) for each resting block."""
    starts = mne.find_events(raw, stim_channel=start_channel, shortest_event=1)[:, 0]
    ends = mne.find_events(raw, stim_channel=end_channel, shortest_event=1)[:, 0]

    assert len(starts) == len(ends), f"{len(starts)} starts vs {len(ends)} ends"

    return list(zip(starts, ends))


def _ast_get_rs_block(raw, block_index, start_channel='STI001', end_channel='STI002'):
    """block_index: 0 = first block, 1 = second block."""
    blocks = _ast_get_rs_events(raw, start_channel, end_channel)
    start_samp, end_samp = blocks[block_index]

    tmin = (start_samp - raw.first_samp) / raw.info['sfreq']
    tmax = (end_samp - raw.first_samp) / raw.info['sfreq']

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