import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import parallel_config, parallel_backend
from mne.io.constants import FIFF
import subprocess
import numpy as np
import logging
import re as re
import shutil
import time
import mne
import joblib
import glob
import json
import pandas as pd
import os

logger = logging.getLogger(__name__)


def run_recon_freesurfer(
    freesurfer_home: str,
    subjects_dir: str,
    license_path: str,
    subject_id: str,
    mri_path: str,
):
    """
    Run FreeSurfer's `recon-all` pipeline on a subject's MRI.

    Sets the required FreeSurfer environment variables and invokes
    `recon-all -all` via subprocess to perform full cortical
    reconstruction for a single subject.

    Parameters
    ----------
    freesurfer_home : str
        Path to the FreeSurfer installation directory.
    subjects_dir : str
        Path to the FreeSurfer SUBJECTS_DIR where output will be stored.
    license_path : str
        Path to the FreeSurfer license file.
    subject_id : str
        Identifier for the subject; used to name the output folder.
    mri_path : str
        Path to the input MRI volume (e.g., T1-weighted image).

    Returns
    -------
    None
    """
    env = os.environ.copy()
    env["FREESURFER_HOME"] = freesurfer_home
    env["SUBJECTS_DIR"] = subjects_dir
    env["FS_LICENSE"] = license_path

    # Path to FreeSurfer setup script
    setup_script = os.path.join(freesurfer_home, "SetUpFreeSurfer.sh")

    # Construct command
    full_command = f"bash -c 'source {setup_script} && recon-all -i {mri_path} -s {subject_id} -all'"

    # Run the command
    process = subprocess.run(full_command, shell=True, env=env)

    if process.returncode == 0:
        print("recon-all completed successfully.")
    else:
        print(f"recon-all failed with exit code {process.returncode}.")


def set_freesurfer_paths(
    freesurfer_home: str,
    subjects_dir: str,
    license_path: str,
):
    """
    Set FreeSurfer-related environment variables for the current process.

    Parameters
    ----------
    freesurfer_home : str
        Path to the FreeSurfer installation directory. Also prepended
        to the system `PATH`.
    subjects_dir : str
        Path to the FreeSurfer SUBJECTS_DIR.
    license_path : str
        Path to the FreeSurfer license file.

    Returns
    -------
    None
    """

    os.environ["FREESURFER_HOME"] = freesurfer_home
    os.environ["PATH"] = os.environ["FREESURFER_HOME"] + "/bin:" + os.environ["PATH"]
    os.environ["SUBJECTS_DIR"] = subjects_dir
    os.environ["FS_LICENSE"] = license_path


def check_freesurfer():
    """
    Locate the FreeSurfer installation and configure environment variables.

    This function attempts to automatically detect the installation path of
    FreeSurfer by first locating the `recon-all` executable in the system PATH.
    If that fails, it checks a set of common installation directories.

    If found, it verifies that a valid license file (`license.txt`) exists
    in the expected directory and sets the necessary environment variables:
    `FREESURFER_HOME` and `FREESURFER_LICENSE`.

    Raises
    ------
    RuntimeError
        If FreeSurfer cannot be found or the license file is missing. The error
        message will include guidance for installing FreeSurfer or manually
        setting the environment variables.

    Returns
    -------
    freesurfer_home : str
        The absolute path to the FreeSurfer installation directory.

    Examples
    --------
    >>> fs_home = find_freesurfer()
    >>> print(f"FreeSurfer found at: {fs_home}")
    """
    ...
    # Try to locate recon-all
    env = os.environ.copy()
    result = subprocess.run(
        ["which", "recon-all"], capture_output=True, text=True, env=env
    )
    recon_path = result.stdout.strip()

    if recon_path:
        freesurfer_home = os.path.abspath(
            os.path.join(os.path.dirname(recon_path), "..")
        )
    else:
        # Try common install paths
        possible_paths = [
            "/opt/freesurfer",
            os.path.expanduser("~/software/freesurfer"),
            os.path.expanduser("~/freesurfer"),
        ]
        freesurfer_home = next(
            (
                p
                for p in possible_paths
                if os.path.exists(os.path.join(p, "SetUpFreeSurfer.sh"))
            ),
            None,
        )

    if not freesurfer_home:
        error_msg = (
            "FreeSurfer not found. Please install it (https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall)"
            " or set FREESURFER_HOME manually if it has already been installed."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Check license file
    license_path = os.path.join(freesurfer_home, "license.txt")
    if not os.path.exists(license_path):
        error_msg = f"FreeSurfer license not found at {license_path}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    os.environ["FREESURFER_LICENSE"] = license_path

    return None


def max_consecutive_ratio(nums):
    """
    Find the maximum absolute ratio between consecutive non-zero elements in a list.

    Parameters
    ----------
    nums : list or array-like of float or int
        A sequence of numerical values. Elements may include zeros,
        which are skipped to avoid division by zero.

    Returns
    -------
    max_index : int
        The index of the first element in the pair with the maximum
        absolute ratio (i.e., the ratio is computed as `abs(nums[i] / nums[i+1])`).
    max_ratio : float
        The maximum absolute ratio found between consecutive non-zero elements.

    Notes
    -----
    - If either element in a consecutive pair is zero, that pair is skipped.
    - The function compares `abs(a / b)` for each valid consecutive pair `(a, b)`.
    - If all elements are zero or no valid pair is found, the function returns the default `(0, 0)`.

    Examples
    --------
    >>> max_consecutive_ratio([2, 4, 1, 8])
    (2, 8.0)

    >>> max_consecutive_ratio([0, 5, 0, 10])
    (1, 0.5)
    """

    max_ratio = 0
    max_index = 0

    for i in range(len(nums) - 1):
        a, b = nums[i], nums[i + 1]
        if a == 0 or b == 0:
            continue  # Skip to avoid division by zero
        ratio = abs(a / b)
        if ratio > max_ratio:
            max_ratio = ratio
            max_index = i

    return max_index, max_ratio


def rank_based_quality_control(
    data_cov,
    info,
    subject,
    figures_path,
    exclude=[],
    qc_ignore=[],
):
    """
    Perform rank-based quality control on a covariance matrix using singular value spectrum analysis.

    This function estimates the rank of the covariance matrix and compares it to the largest
    "cliff" (drop) in the singular value spectrum. If the estimated rank exceeds the index of the
    largest drop, it raises an exception and saves a diagnostic figure unless the subject is
    explicitly listed in `qc_ignore`.

    Parameters
    ----------
    data_cov : instance of mne.Covariance
        The data covariance matrix to be quality-checked.

    info : instance of mne.Info
        The measurement info dictionary containing metadata about the channels.

    subject : str
        Identifier of the subject, used for logging and saving figures.

    figures_path : str
        Path to the directory where quality control figures should be saved.

    exclude : list of str, optional
        List of channel names to exclude from the analysis (e.g., bad channels).

    qc_ignore : list of str, optional
        List of subject identifiers to ignore QC errors for (i.e., suppress exception and allow
        continuing despite failing the QC check).

    Raises
    ------
    Exception
        If the estimated rank exceeds the index of the largest drop in the singular value spectrum
        and the subject is not in `qc_ignore`.

    Returns
    -------
    None

    Notes
    -----
    - Singular values are regularized by replacing non-positive values to avoid numerical instability.
    - The largest consecutive drop in the spectrum is used as a heuristic for determining the intrinsic dimensionality.
    - Based on implementation and utilities from the MNE-Python package.

    Examples
    --------
    >>> rank_based_quality_control(data_cov, info, 'sub-01', '/path/to/figures')

    If a QC issue is found and `subject` is not in `qc_ignore`, a diagnostic plot will be saved and an exception raised.
    """

    os.makedirs(os.path.join(figures_path, "QC_results"), exist_ok=True)

    # This lines of codes were copied from MNE:
    # https://mne.tools/stable/index.html
    # ====================================================================
    info, C, ch_names, idx_names = mne.viz.misc._index_info_cov(
        info, data_cov, exclude=exclude
    )

    for k, (idx, name, unit, scaling, key) in enumerate(idx_names):
        this_C = C[idx][:, idx]
        s = np.linalg.svd(this_C, compute_uv=False)

        this_C = mne.cov.Covariance(
            this_C, [info["ch_names"][ii] for ii in idx], [], [], 0
        )
        this_info = mne._fiff.pick.pick_info(info, idx)
        with this_info._unlock():
            this_info["projs"] = []
        this_rank = mne.rank.compute_rank(this_C, info=this_info)

        this_rank = this_rank[key]

        s[s <= 0] = 1e-10 * s[s > 0].min()
        s = np.sqrt(s) * scaling
        # ====================================================================

        diffs = np.diff(s)
        abs_diffs = abs(diffs)

        eigen_idx, _ = max_consecutive_ratio(abs_diffs)

        logger.info(f"The estimated rank of data_cov: {this_rank}")
        logger.info(
            f"The largest drop (cliff) in the singular value spectrum: {eigen_idx}"
        )
        if not this_rank > eigen_idx:

            if subject in qc_ignore:
                pass

            else:

                _, fig = data_cov.plot(info=info)
                fig.savefig(
                    os.path.join(figures_path, "QC_results", f"{subject}_QC_rank.png")
                )

                error_msg = (
                    f"There seems to be a problem with the covariance matrix for {name}. The estimated "
                    "rank is higher than the largest drop (cliff) in the singular value spectrum. "
                    "The spectrum for this subject will be saved—please review it. If you determine "
                    "there is no issue, you can add the subject ID to the qc_ignore argument."
                )
                logger.error(error_msg)
                raise Exception(error_msg)

    return


def save_coreg_screenshots(
    info, trans, subject, subjects_dir, out_dir, participant_id, **kwargs
):
    import os
    import mne
    import pyvista

    os.makedirs(out_dir, exist_ok=True)

    try:
        pyvista.OFF_SCREEN = True
        mne.viz.set_3d_backend("pyvistaqt")

        fig = mne.viz.plot_alignment(
            info,
            trans=trans,
            subject=subject,
            subjects_dir=subjects_dir,
            surfaces="head",
            dig=True,
            eeg=[],
            meg="sensors",
            show_axes=True,
            coord_frame="meg",
            mri_fiducials=kwargs.get("corregistration_fiducials", "estimated"),
        )
        plotter = fig.plotter
        plotter.off_screen = True

        views = {
            "front": dict(azimuth=90, elevation=90),
            "lateral_left": dict(azimuth=180, elevation=90),
            "lateral_right": dict(azimuth=0, elevation=90),
        }

        for view_name, angles in views.items():
            mne.viz.set_3d_view(fig, **angles, distance=0.6)
            out_path = os.path.join(out_dir, f"{participant_id}_coreg_{view_name}.png")
            plotter.screenshot(out_path)
            logger.info(f"Saved coregistration screenshot: {out_path}")

        mne.viz.close_3d_figure(fig)

    except Exception as e:
        logger.warning(
            f"Coregistration screenshot failed for {participant_id}; "
            f"skipping QC image, pipeline will continue. Error: {e}"
        )


def corregistration(
    data, subject, subjects_dir, participant_id, qc_out_dir, plot_3d, **kwargs
):
    """
    Coregister MEG data to MRI, with optional scaling to a template MRI.

    Same as before, but if `coregisteration_scale_mode` is set (e.g. "uniform"),
    a scale factor is estimated during fitting and a physically scaled copy of
    the subject's MRI is written to `subjects_dir` as `{subject}_scaled`.
    Downstream steps (BEM, source space, parcellation) must use the returned
    `fit_subject` name, not the original template name.

    Returns
    -------
    coreg : mne.coreg.Coregistration
    fit_subject : str
        The subject name to use for all subsequent anatomy-dependent steps
        (equals `subject` if no scaling was applied, else `f"{subject}_scaled"`).
    """

    if not os.path.exists(
        os.path.join(subjects_dir, subject, "bem", "inner_skull.surf")
    ) or kwargs.get("force_new_watershed_bem"):

        logger.info("bem surface was not found; Creating a bem surface for the subject")

        mne.bem.make_watershed_bem(
            subject=subject,
            subjects_dir=subjects_dir,
            overwrite=True,
            gcaatlas=kwargs.get("gcaatlas", True),
            volume="T1",
            preflood=kwargs.get("preflood", None),
        )

    coreg = mne.coreg.Coregistration(
        data.info,
        subject=subject,
        subjects_dir=subjects_dir,
        fiducials=kwargs.get("corregistration_fiducials", "estimated"),
    )

    scale_mode = kwargs.get(
        "coregisteration_scale_mode", None
    )  # None, "uniform", "3-axis"
    if scale_mode:
        coreg.set_scale_mode(scale_mode)

    coreg.fit_fiducials()

    coreg.fit_icp(
        n_iterations=kwargs.get("coregisteration_initial_n_iterations", 6),
        nasion_weight=kwargs.get("coregisteration_initial_nasion_weight", 2.0),
        verbose=True,
    )

    coreg.omit_head_shape_points(
        distance=kwargs.get("coregisteration_distance_thr", 5.0 / 1000)
    )

    coreg.fit_icp(
        n_iterations=kwargs.get("coregisteration_final_n_iterations", 20),
        nasion_weight=kwargs.get("coregisteration_final_nasion_weight", 10.0),
        verbose=True,
    )

    distance_head_mri = coreg.compute_dig_mri_distances()
    logger.info(
        f"Average and STD distance between head shape points and MRI surface: "
        f"{np.mean(distance_head_mri)} and {np.std(distance_head_mri)}"
    )

    fit_subject = subject
    if scale_mode:
        # scale_id = participant_id or subject
        scaled_subject = f"{participant_id}_scaled"
        # scaled_bem_exists = os.path.exists(
        #     os.path.join(subjects_dir, scaled_subject, "bem", "inner_skull.surf")
        # )

        # if not scaled_bem_exists:
        logger.info(f"Estimated MRI scale factor: {coreg.scale}")
        mne.scale_mri(
            subject_from=subject,
            subject_to=scaled_subject,
            scale=coreg.scale,
            subjects_dir=subjects_dir,
            overwrite=True,
            labels=True,
            skip_fiducials=True,
        )
        logger.info(f"Scaled MRI subject written: {scaled_subject}")

        # TODO: is it necessary to make a new watershed mode after scaling?
        # if kwargs.get("force_new_watershed_bem", False):
        #     pass  
            # mne.bem.make_watershed_bem(
            #     subject=scaled_subject,
            #     subjects_dir=subjects_dir,
            #     overwrite=True,
            #     gcaatlas=kwargs.get("gcaatlas", True),
            #     volume="T1",
            #     preflood=kwargs.get("preflood", None),
            # )
            # logger.info(
            #     f"Watershed BEM regenerated for scaled subject: {scaled_subject}"
            # )
        # else:
        #     logger.info(f"Using existing scaled subject: {scaled_subject}")

        fit_subject = scaled_subject

    if kwargs.get("take_screenshot_of_coregisteration", True):
        save_coreg_screenshots(
            info=data.info,
            trans=coreg.trans,
            subject=fit_subject,
            subjects_dir=subjects_dir,
            out_dir=qc_out_dir,
            participant_id=fit_subject,
            **kwargs,
        )

    logger.info("Automatic coregisteration is done!")

    return coreg, fit_subject


def forward_solution(
    subject,
    subjects_dir,
    data,
    transformation_matrix,
    conductivity,
    source_space,
    which_sensor_dict,
    **kwargs,
):
    """
    Compute the forward solution (lead field matrix) for MEG/EEG source localization.

    This function computes a forward model using a subject's MRI, a coregistration transform,
    a BEM model, and a source space (surface or volumetric). The result is a lead field matrix
    used for inverse modeling in source localization.

    Parameters
    ----------
    subject : str
        The subject name as used in FreeSurfer.
    subjects_dir : str or Path
        Path to the FreeSurfer SUBJECTS_DIR containing all subject MRI folders.
    data:
        To be written
    transformation_matrix : str, Path, or dict
        Transformation between head and MRI coordinates, typically from coregistration.
    conductivity : tuple of float
        Conductivity values for BEM layers, e.g., (0.3,) for a single-layer model.
    source_space : str
        Type of source space to create. Must be either "surface" or "volumetric".

    **kwargs : dict, optional
        Optional parameters to control source space and forward model configuration:

        - source_space_spacing : str, default="oct6"
            Spacing for surface source space (e.g., "oct6", "ico5").
        - source_space_add_dist : str, default="patch"
            Whether to add patch information to the surface source space.
        - source_space_spacing : int, default=5
            BEM surface tessellation grade (higher = finer mesh).
        - meg : bool, default=True
            Whether to include MEG channels in the forward model.
        - eeg : bool, default=False
            Whether to include EEG channels in the forward model.
        - forward_mindist : float, default=5.0
            Minimum distance in millimeters from source points to inner skull surface.
        - n_jobs : int, default=-1
            Number of parallel jobs to use (set -1 to use all available cores).

    Returns
    -------
    lead_field_matrix : mne.Forward
        The computed forward solution object, containing the lead field matrix.

    Notes
    -----
    - FreeSurfer anatomical reconstructions (recon-all) and BEM surfaces must already exist.
    - For volumetric source spaces, the inner skull surface must be present at:
      subjects_dir/subject/bem/inner_skull.surf
    """

    logger.info(f"Setting up a {source_space} source space")
    # source space
    if source_space == "surface":
        src = mne.setup_source_space(
            subject=subject,
            subjects_dir=subjects_dir,
            spacing=kwargs.get("source_space_spacing", "ico6"),
            add_dist=kwargs.get("source_space_add_dist", "patch"),
            n_jobs=kwargs.get("n_jobs", 1),
        )

    elif source_space == "volumetric":
        src = mne.setup_volume_source_space(
            subject=subject,
            subjects_dir=subjects_dir,
            surface=Path(subjects_dir) / subject / "bem" / "inner_skull.surf",
            add_interpolator=True,
            n_jobs=kwargs.get("n_jobs", 1),
        )

    # forward model
    bem_model = mne.make_bem_model(
        subject=subject,
        ico=kwargs.get("source_space_spacing_number", 6),
        conductivity=conductivity,
        subjects_dir=subjects_dir,
    )

    bem = mne.make_bem_solution(bem_model)
    logger.info(
        f"{source_space} BEM model with {len(conductivity)} layer/s was constructed."
    )

    lead_field_matrix = mne.make_forward_solution(
        data.info,
        trans=transformation_matrix,
        src=src,
        bem=bem,
        meg=bool(
            which_sensor_dict.get("meg")
            or which_sensor_dict.get("grad")
            or which_sensor_dict.get("mag")
        ),
        eeg=which_sensor_dict.get("eeg", False),
        mindist=kwargs.get("forward_mindist", 5.0),
        n_jobs=kwargs.get("n_jobs", 1),
        verbose=True,
        ignore_ref=kwargs.get("source_localization_ignore_ref", True),
    )
    logger.info("Lead field matrix was estimated.")

    # You need to use src for further analysis (morphing) from fwd since vertices can be excluded
    # due to their proximity to inner skull surface
    return lead_field_matrix, lead_field_matrix["src"]


def inverse_solution(
    subject,
    data,
    segments,
    fwd,
    inverse_operator,
    figures_path,
    which_sensor_dict,
    source_space=None,
    empty_room_recording=None,
    qc_ignore=[],
    **kwargs,
):
    """
    Compute the inverse solution using an LCMV beamformer.

    This function estimates the source activity from MEG data using an inverse solution.
    It supports computing noise covariance from empty-room recordings and data covariance
    from the MEG signal. The LCMV beamformer is currently the only supported inverse method.

    Parameters
    ----------
    data : mne.io.Raw
        The preprocessed M/EEG data to apply the inverse solution on.
    fwd : mne.Forward
        The forward solution (lead field matrix).
    inverse_operator : str
        The type of inverse method to use. Currently only supports "lcmv".
    source_space : str
        Type of source space to create. Must be either "surface" or "volumetric".
    empty_room_recording : mne.io.Raw
        Empty-room MEG recording used to estimate noise covariance.
        If None, no noise covariance will be used.


    **kwargs : dict, optional
        Optional parameters to control the inverse solution:

        - covariance_method : str, default="empirical"
            Method used to estimate the covariance matrices. Should be one of
            "empirical", "shrunk", or "oas".
        - inverse_regularization_value : float, default=0.05
            Regularization parameter added to stabilize the inverse computation.
        - beamformer_pick_ori : str, default="max-power"
            Orientation selection method for the LCMV filter. Options: "max-power", "normal", or None.
        - beamformer_weight_norm : str or None, default="unit-noise-gain"
            Type of weight normalization to apply. Options include "unit-noise-gain", "nai", or None.

    Returns
    -------
    stc : mne.SourceEstimate
        The source time course estimate resulting from the inverse solution.

    Notes
    -----
    - This function currently only supports LCMV beamformer inverse methods.
    - It assumes the forward model is already computed and passed as `fwd`.
    - Noise covariance can be estimated from an empty-room recording if provided.
    """
    # if tSSS has already been applied, return the rank in info
    if check_tsss(meg_data=data):
        segments_rank = mne.compute_rank(segments, rank="info")
    else:
        # this estimates the rank after scaling
        segments_rank = mne.compute_rank(segments, rank=None)

    if empty_room_recording is not None:
        noise_cov = mne.compute_raw_covariance(
            empty_room_recording,
            method=kwargs.get("covariance_method", "empirical"),
            n_jobs=kwargs.get("n_jobs", 1),
        )  # TODO: change to epoch later

        logger.info(
            "Noise covariance was calculated from  empty room recordings. This will be used to pre-whiten"
            "the data"
        )

        noise_rank = mne.compute_rank(empty_room_recording)

        mag_in_data = bool("mag" in segments_rank)
        grad_in_data = bool("grad" in segments_rank)
        if mag_in_data:
            if segments_rank["mag"] < noise_rank["mag"]:
                noise_rank["mag"] = segments_rank["mag"]
        if grad_in_data:
            if segments_rank["grad"] < noise_rank["grad"]:
                noise_rank["grad"] = segments_rank["grad"]

        # According to MNE: When a noise covariance is used for whitening,
        # this should reflect the rank of that covariance, otherwise
        # amplification of noise components can occur in whitening
        lcmv_rank = noise_rank.copy()

    else:
        # If both MAG and Grad are present, having a noise cov is necessary
        # to scale the data. If empty room recording is not available,
        # make_ad_hoc_cov make a diagonal covariance matrix where the
        # diagonals represent channel wise variance and off-diagonals are zero.
        # The default noise values are 5 fT/cm, 20 fT for gradiometers, magnetometers
        logger.info(
            "Empty room recording is not available. Therefore, a diagonal covariance matrix where the"
            " diagonals represent channel wise variance and off-diagonals are zero is used to whitten the data."
        )

        noise_cov = mne.make_ad_hoc_cov(
            info=segments.info, std=kwargs.get("ad_hoc_cov_std", None)
        )
        lcmv_rank = segments_rank.copy()

    if inverse_operator == "lcmv":
        logger.info(
            f"Solving the inverse problem using {inverse_operator} algorithm. "
            f"A regularization of {kwargs.get('inverse_regularization_value', 0.05)} will be used "
            f"to shift the matrix so it can be invertible. Furthermore, we will use {kwargs.get('beamformer_pick_ori', 'max-power')} for `pick_ori`."
        )

        # compute segments covaraince
        segments_cov = mne.compute_covariance(
            segments,
            method=kwargs.get("covariance_method", "empirical"),
            rank=lcmv_rank,  # TODO: this should be removed
            n_jobs=kwargs.get("n_jobs", 1),
        )

        if not kwargs.get("beamforme_depth") and source_space == "volumetric":
            error_msg = (
                "If you want to use volumetric source space (interested in deeper sources),"
                " please define beamforme_depth as positive float number, i.e., 0.8. This is used to address"
                " the center of head bias."
            )
            logger.error(error_msg)
            raise Exception(error_msg)

        # rank_based_quality_control(
        #     data_cov=data_cov,
        #     info=data.info,
        #     subject=subject,
        #     figures_path=figures_path,
        #     exclude=[], #TODO
        #     qc_ignore=qc_ignore)

        # _, cond_before, cond_after = regularized_cov_condition(data_cov.data,
        #         shrinkage=0.05,
        #         diag_scale='auto')
        # logger.info(f"Condition number of data covariance before regularization: {cond_before}")
        # logger.info(f"Condition number of data covariance After regularization: {cond_after}")

        filters = mne.beamformer.make_lcmv(
            segments.info,
            forward=fwd,
            data_cov=segments_cov,
            noise_cov=noise_cov,
            reg=kwargs.get(
                "inverse_regularization_value", 0.05
            ),  # for regularization (shifting the matrix)
            pick_ori=kwargs.get("beamformer_pick_ori", "max-power"),
            weight_norm=kwargs.get("beamformer_weight_norm", "unit-noise-gain"),
            rank=lcmv_rank,  # if er_recording available ==> noise rank, otherwise, data rank
            depth=kwargs.get("beamforme_depth", None),
        )

    stc = mne.beamformer.apply_lcmv_epochs(segments, filters=filters)

    logger.info("Source estimate is done!")
    return stc


def morph_stc(
    subject,
    subject_to,
    subjects_dir,
    stc,
    src_from,
    source_space,
    plot_3d=False,
    **kwargs,
):
    """
    Morph a source estimate (STC) to a different subject's brain anatomy.

    This function morphs a source estimate from a given subject
    to a target subject (e.g., 'fsaverage') using MNE's spherical morphing.
    Optionally, it visualizes the morphed data using an interactive 3D brain viewer.

    Parameters
    ----------
    subject : str
        Name of the subject from which the source estimate was computed.
    subject_to : str
        Name of the subject to which the source estimate should be morphed.
    subjects_dir : str
        Path to the FreeSurfer subjects directory.
    stc : mne.SourceEstimate
        The source estimate to be morphed, computed for `subject`.
    src_from : instance of mne.SourceSpaces
        The source space in which `stc` was computed.
    source_space : str
        Type of source space to set up for the target subject. Must be
        either "surface" or "volumetric".
    plot_3d : bool, optional
        Whether to display the morphed source estimate using MNE's interactive 3D plotter.
        Default is False.
    **kwargs : dict
        Additional keyword arguments passed to `mne.compute_source_morph()`, allowing
        customization of the morphing process. For example:
            spacing : str | int
                The spacing to use for morphing (default is 5 in this function).
            smooth : int
                Number of smoothing steps.
            warn : bool
                Whether to emit warnings during morphing.
            etc.

    Returns
    -------
    stc_fsaverage : mne.SourceEstimate
        The morphed source estimate in the space of `subject_to`.
    src_morph_to : instance of mne.SourceSpaces
        The source space constructed for `subject_to`, used as the
        morph target.

    Notes
    -----
    - The `src_from` argument is included for completeness but not
      directly used in the morph computation itself.
    - For volumetric source spaces, the inner skull surface for
      `subject_to` is generated via `mne.bem.make_watershed_bem` if
      not already present.
    """
    logger.info("Morphing the estimated source data onto a common space")

    if source_space == "surface":

        src_morph_to = mne.setup_source_space(
            subject=subject_to,
            subjects_dir=subjects_dir,
            spacing=kwargs.get("source_space_spacing", "ico6"),
            add_dist=kwargs.get("source_space_add_dist", "patch"),
            n_jobs=kwargs.get("n_jobs", 1),
        )

    elif source_space == "volumetric":  # TODO

        inner_skull_path = Path(subjects_dir) / subject_to / "bem" / "inner_skull.surf"
        if not os.path.exists(inner_skull_path):
            mne.bem.make_watershed_bem(
                subject=subject_to,
                subjects_dir=subjects_dir,
                overwrite=True,
                gcaatlas=kwargs.get("gcaatlas", True),
                volume="T1",  # TODO: this should be a data specific info.
                preflood=kwargs.get("preflood", None),
            )

        src_morph_to = mne.setup_volume_source_space(
            subject=subject_to,
            subjects_dir=subjects_dir,
            surface=inner_skull_path,
            add_interpolator=True,
            n_jobs=kwargs.get("n_jobs", 1),
        )

    with parallel_backend("threading"):
        with parallel_config(n_jobs=1):
            morph = mne.compute_source_morph(
                src_from,
                subject_from=subject,
                src_to=src_morph_to,
                subject_to=subject_to,
                subjects_dir=subjects_dir,
                spacing=kwargs.get("source_space_spacing_number", 6),
            )

            logger.info("Starting the morphing process")
            start_time = time.time()
            stc_fsaverage = morph.apply(stc)

    if plot_3d:
        brain = stc_fsaverage.plot(
            subjects_dir=subjects_dir,
            clim=dict(kind="value", lims=[3, 6, 9]),
            smoothing_steps=7,
        )

    elapsed = time.time() - start_time
    logger.info(f"Morphing complete. Elapsed time: {elapsed:.2f} seconds.")
    return stc_fsaverage, src_morph_to


def parcellate(subject, subjects_dir, stc, src, source_space, **kwargs):
    """
    Parcellate a morphed source estimate into anatomical regions.

    This function extracts label-wise time courses from a source estimate that has been
    morphed to a common brain (e.g., 'fsaverage') using a cortical parcellation atlas.
    It uses MNE's `extract_label_time_course()` to summarize activity within each
    anatomical region defined by a chosen parcellation.

    Parameters
    ----------
    subject : str
        The subject name to use for anatomical labels and source space setup (typically 'fsaverage').
    subjects_dir : str
        Path to the FreeSurfer subjects directory containing anatomical reconstructions.
    stc : mne.SourceEstimate | list of SourceEstimate
        The source estimate(s) already morphed to `subject`. Can be a single STC or a list.
    source_space : str
        Type of source space to create. Must be either "surface" or "volumetric".
    **kwargs : dict
        Optional keyword arguments to customize the parcellation process:

        parcellation_parc : str, default='aparc.a2009s'
            The cortical parcellation scheme to use (e.g., 'aparc', 'aparc.a2009s', 'HCPMMP1').

        source_space_spacing : str | int, default='ico6'
            The spacing to use when setting up the source space (e.g., 'ico5', 'ico6', 'oct6').

        source_space_add_dist : bool | str, default='patch'
            Whether to compute patch information (distance matrix) in the source space.

        parcellation_mode : str, default='mean_flip'
            The method used to extract the label time course.
            Options: 'mean', 'mean_flip', 'pca_flip', 'max', etc.

    Returns
    -------
    parcelled_stc : ndarray, shape (n_labels, n_times)
        The time series for each anatomical label. Each row corresponds to one label, each column to a time point.

    Notes
    -----
    - This function assumes the source estimate is already morphed to the desired subject (e.g., 'fsaverage').
    - The source space is reconstructed internally for label extraction and does not need to match the original STC.
    - The default parcellation ('aparc.a2009s') provides 148 cortical labels (74 per hemisphere).
    """
    logger.info(
        f"Parcellating morphed source estimates using {kwargs.get('parcellation_parc', 'aparc.a2009s')} atlas."
    )

    if not os.path.exists(os.path.join(subjects_dir, subject)) and kwargs.get(
        "apply_morphing"
    ):
        mne.datasets.fetch_fsaverage()

    if source_space == "surface":
        labels = mne.read_labels_from_annot(
            subject=subject,
            subjects_dir=subjects_dir,
            parc=kwargs.get("parcellation_parc", "aparc.a2009s"),
            annot_fname=kwargs.get("parcellation_annot_fname", None),
        )
        labels_list = [label.name for label in labels]

    elif source_space == "volumetric":
        parc = kwargs.get("parcellation_parc", "aparc.a2009s")
        labels = os.path.join(subjects_dir, subject, "mri", f"{parc}+aseg.mgz")
        labels_list = mne.get_volume_labels_from_aseg(
            mgz_fname=labels, return_colors=False
        )

    else:
        error_msg = "Source space model is not detected. Source splace must be either 'surface' or 'volumetric'."
        logger.error(error_msg)
        raise ValueError(error_msg)

    parcelled_stc = mne.extract_label_time_course(
        stcs=stc,
        labels=labels,
        src=src,
        mode=kwargs.get("parcellation_mode", "auto"),
        return_generator=False,
    )

    logger.info("Parcellation is finised!")

    return parcelled_stc, labels_list


def source_localization(
    project_dir,
    subject,
    subjects_dir,
    subject_to,
    data,
    segments,
    figures_path,
    which_sensor_dict,
    source_space="surface",
    conductivity=(0.3,),
    inverse_operator="lcmv",
    plot_3d=False,
    qc_ignore=[],
    empty_room_recording=None,
    **kwargs,
):
    """
    Perform full source localization pipeline and parcellation of M/EEG data.

    This function runs a complete source localization workflow including:
    coregistration, forward modeling, inverse solution computation, morphing the
    source estimate to a standard brain (e.g., 'fsaverage'), and parcellating
    the morphed source estimate using anatomical labels.

    Parameters
    ----------
    subject : str
        The name of the subject (must correspond to a FreeSurfer reconstruction).
    subjects_dir : str
        Path to the FreeSurfer `SUBJECTS_DIR` containing anatomical reconstructions.
    subject_to : str
        Target subject for morphing the source estimate (typically 'fsaverage').
    data : instance of mne.io.Raw | Epochs | Evoked
        The data to localize. Must be preprocessed and contain sensor locations.
    freesurfer_path : str
        Path to the FreeSurfer installation (used to set environment variables).
    freesurfer_license_path : str
        Path to the FreeSurfer license file.
    source_space : str, default='surface'
        Type of source space to use ('surface' or 'volume').
    conductivity : tuple of float, default=(0.3,)
        Conductivity values for the BEM model (e.g., one value for 1-layer, three for 3-layer).
    inverse_operator : str, default='lcmv'
        Method to compute the inverse solution. Options: 'lcmv', 'dspm', 'mne', etc.
    empty_room_recording: mne.io.Raw
        empty_room_recording
    Returns
    -------
    stc : ndarray, shape (n_labels, n_times)
        The parcellated source time series. Each row corresponds to a brain region, and each column to a time point.

    Notes
    -----
    - This function assumes all anatomical preprocessing (e.g., `recon-all`) has been done for `subject`.
    - Morphing is performed using spherical surface morphing to a target subject (e.g., fsaverage).
    - Only surface-based source localization is currently supported.
    - The BEM model is generated internally using provided conductivity values.
    """

    # set_freesurfer_paths(
    #     freesurfer_home=freesurfer_path,
    #     subjects_dir=subjects_dir,
    #     license_path=freesurfer_license_path,
    # )

    # check_freesurfer()
    # Set FreeSurfer environment variables so all subprocess calls can find the license
    if kwargs.get("freesurfer_home"):
        os.environ["FREESURFER_HOME"] = kwargs.get("freesurfer_home")
        os.environ["PATH"] = (
            kwargs.get("freesurfer_home") + "/bin:" + os.environ["PATH"]
        )
    if kwargs.get("freesurfer_license"):
        os.environ["FS_LICENSE"] = kwargs.get("freesurfer_license")

    if kwargs.get("which_sensor", "meg") in ["meg", "grad", "mag"]:
        new_dig = [d for d in data.info["dig"] if d["kind"] != FIFF.FIFFV_POINT_EEG]
        with data.info._unlock():
            data.info["dig"] = new_dig

    participant_id = subject
    if kwargs.get("apply_mri_template"):
        subject, subjects_dir = prepare_template(
            subject=subject, project_dir=project_dir, **kwargs
        )

    coreg, subject = corregistration(
        data=data,
        subject=subject,
        subjects_dir=subjects_dir,
        participant_id=participant_id,
        qc_out_dir=os.path.join(project_dir, "Saved_outputs", "coregistration_QC"),
        plot_3d=plot_3d,
        **kwargs,
    )

    fwd, src = forward_solution(
        subject=subject,
        subjects_dir=subjects_dir,
        data=data,
        transformation_matrix=coreg.trans,
        conductivity=conductivity,
        source_space=source_space,
        which_sensor_dict=which_sensor_dict,
        **kwargs,
    )

    del coreg

    stc = inverse_solution(
        subject=subject,
        data=data,
        segments=segments,
        fwd=fwd,
        inverse_operator=inverse_operator,
        source_space=source_space,
        empty_room_recording=empty_room_recording,
        figures_path=figures_path,
        qc_ignore=qc_ignore,
        which_sensor_dict=which_sensor_dict,
        **kwargs,
    )

    del fwd

    # using the variable apply_morphing, you can choose whether you need
    # morphing stc to a common source space or not
    if kwargs.get("apply_morphing", False):
        stc, src = morph_stc(
            subject=subject,
            subject_to=subject_to,
            subjects_dir=subjects_dir,
            stc=stc,
            src_from=src,
            source_space=source_space,
            plot_3d=plot_3d,
            **kwargs,
        )
        subject = subject_to

    stc, labels = parcellate(
        subject=subject,
        subjects_dir=subjects_dir,
        stc=stc,
        src=src,
        source_space=source_space,
        **kwargs,
    )

    logger.info("Done; congrats! ")

    return stc, labels


def numpy_to_mne_epoch(stc, labels, ch_name, sampling_rate):
    """
    Convert a parcellated source estimate into an MNE Epoch object.

    This function wraps a 2D NumPy array representing parcellated source time series
    into an `mne.io.RawArray` using anatomical labels and sampling frequency information.

    Parameters
    ----------
    stc : ndarray, shape (n_labels, n_times)
        The source time courses for each label (typically from `extract_label_time_course`).
        Each row corresponds to one anatomical region, and each column to a time point.
    labels : list of mne.Label or mne.VolumeLabel
        The list of anatomical labels used to extract the time courses. Must match the number of rows in `stc`.
    ch_name : str
        The MNE channel type to assign to all channels (e.g., 'misc', 'eeg', 'ecog').
        Must be one of the types supported by `mne.create_info`.
    sampling_rate : float
        The sampling frequency (in Hz) of the source time series.

    Returns
    -------
    raw_parc : mne.io.RawArray
        The MNE Raw object containing the parcellated source estimate as virtual channels.

    Notes
    -----
    - This is commonly used to wrap parcellated source activity into a Raw object
      so it can be saved, plotted, or processed using MNE’s standard pipeline.
    - Ensure that `ch_name` is a valid MNE channel type, such as `'misc'` or `'eeg'`.

    Examples
    --------
    >>> raw = numpy_to_mne_Epoch(parcelled_stc, labels, ch_name='misc', sampling_rate=1000)
    >>> raw.plot()
    """
    ch_types = [ch_name] * len(labels)
    info = mne.create_info(ch_names=labels, sfreq=sampling_rate, ch_types=ch_types)
    epochs = mne.EpochsArray(stc, info)
    return epochs


def regularized_cov_condition(X, shrinkage=0.05, diag_scale="auto"):
    """
    Compute empirical covariance, apply linear shrinkage regularization,
    and return both the regularized covariance and condition numbers.

    Parameters
    ----------
    X : array, shape (n_samples, n_channels)
        Data matrix (zero-mean not required; will be centered internally).
    shrinkage : float
        Shrinkage factor (0 = no regularization, 1 = full diagonal).
    diag_scale : 'auto' or float
        Scaling for identity target. If 'auto', uses mean channel variance.

    Returns
    -------
    C_reg : ndarray
        Regularized covariance matrix.
    cond_before : float
        Condition number of empirical covariance (np.linalg.cond).
    cond_after : float
        Condition number of regularized covariance (np.linalg.cond).
    """
    # Center the data
    Xc = X - X.mean(axis=0, keepdims=True)
    n_samples = Xc.shape[0]

    # Empirical covariance
    C = (Xc.T @ Xc) / (n_samples - 1)

    # Compute target (scaled identity)
    if diag_scale == "auto":
        alpha = np.trace(C) / C.shape[0]
    else:
        alpha = float(diag_scale)
    target = alpha * np.eye(C.shape[0])

    # Regularize covariance
    C_reg = (1 - shrinkage) * C + shrinkage * target

    # Compute condition numbers
    cond_before = np.linalg.cond(C)
    cond_after = np.linalg.cond(C_reg)

    return C_reg, cond_before, cond_after


def check_tsss(meg_data):
    """
    Check if Maxwell filtering (tSSS) was applied to raw/epochs data.

    This inspects the processing history for presence of maxfilter info.

    Parameters
    ----------
    meg_data : mne.io.BaseRaw | mne.Epochs
        The MEG data object.

    Returns
    -------
    bool
        True if tSSS has been applied, False otherwise.
    """
    proc_history = meg_data.info.get("proc_history", [])
    if not proc_history:
        return False
    max_info = proc_history[0].get("max_info", {})
    sss_cal = max_info.get("sss_info", [])
    return len(sss_cal) > 0


def produce_aparc_a2009s_aseg(save_path, freesurfer_home, freesurfer_license):
    """
    Generate the Destrieux (aparc.a2009s) volumetric segmentation for
    every subject in a directory.

    Iterates over subject folders in `save_path` and runs
    `mri_aparc2aseg --a2009s` for each one that does not already have
    an `aparc.a2009s+aseg.mgz` output, skipping those that do.

    Parameters
    ----------
    save_path : str
        Path to the FreeSurfer SUBJECTS_DIR containing subject folders.
    freesurfer_home : str
        Path to the FreeSurfer installation directory.
    freesurfer_license : str
        Path to the FreeSurfer license file.

    Returns
    -------
    None
    """
    for subject in os.listdir(save_path):
        out_path = os.path.join(save_path, subject, "mri", "aparc.a2009s+aseg.mgz")
        if os.path.exists(out_path):
            print(f"Skipping {subject}: aparc.a2009s+aseg.mgz already exists.")
            continue
        env = os.environ.copy()
        env["FREESURFER_HOME"] = freesurfer_home
        env["SUBJECTS_DIR"] = save_path  # <-- this is the fix
        env["FS_LICENSE"] = freesurfer_license
        env["PATH"] = freesurfer_home + "/bin:" + env["PATH"]
        subprocess.run(
            ["mri_aparc2aseg", "--s", subject, "--a2009s"],
            env=env,
            check=True,
        )


def build_template_index(subjects_dir):
    """
    Build an index of available ANTS infant/child MRI templates and
    their corresponding ages in months.

    Scans `subjects_dir` for folders matching the ANTS template naming
    convention and parses each folder name to compute the represented
    age in months.

    Parameters
    ----------
    subjects_dir : str
        Path to the directory containing downloaded ANTS template
        folders.

    Returns
    -------
    index : dict
        Mapping from template folder name to age in months.
    """
    index = {}
    for path in glob.glob(os.path.join(subjects_dir, "ANTS*")):
        name = os.path.basename(path)
        m = re.match(r"ANTS(\d+)-(\d+)(Month|Year)s?3T", name)
        if not m:
            continue
        whole, dec, unit = int(m.group(1)), int(m.group(2)), m.group(3)
        age = whole + dec / 10.0
        if unit == "Year":
            age *= 12
        index[name] = age
    return index


def nearest_template_dir(age_months, subjects_dir):
    """
    Find the pre-downloaded ANTS template closest in age to a target age.

    Parameters
    ----------
    age_months : float
        Target age, in months, to match against available templates.
    subjects_dir : str
        Path to the directory containing downloaded ANTS template
        folders.

    Returns
    -------
    name : str
        Folder name of the nearest-matching template.
    path : str
        Path to `subjects_dir` (the directory containing the template).

    Raises
    ------
    FileNotFoundError
        If no ANTS templates are found in `subjects_dir`.
    """
    index = build_template_index(subjects_dir)
    if not index:
        raise FileNotFoundError(f"No ANTS templates found in {subjects_dir}")
    name = min(index, key=lambda k: abs(index[k] - age_months))
    print(
        f"Nearest template: {name} ({index[name]:.1f} months, requested {age_months:.1f} months)"
    )
    return name, os.path.join(subjects_dir)


def prepare_template(subject, project_dir, **kwargs):
    """
    Select an age-matched anatomical template for a subject and, if
    needed, generate its Destrieux volumetric segmentation.

    Looks up the subject's age from the project's demographic file,
    finds the nearest available ANTS template, and optionally runs
    `produce_aparc_a2009s_aseg` when a volumetric source space with
    the 'aparc.a2009s' parcellation is requested.

    Parameters
    ----------
    subject : str
        Subject identifier, used to look up dataset and demographic
        information.
    project_dir : str
        Path to the project directory containing the
        `Configurations/runner_params.json` file.
    **kwargs : dict, optional
        Additional configuration options, including:

        - SL_source_space : str
            Source space type; triggers segmentation generation when
            'volumetric'.
        - parcellation_parc : str
            Parcellation scheme; triggers segmentation generation when
            'aparc.a2009s'.
        - freesurfer_template_path : str
            Path to the directory of downloaded ANTS templates.
        - freesurfer_home : str
            Path to the FreeSurfer installation directory.
        - freesurfer_license : str
            Path to the FreeSurfer license file.

    Returns
    -------
    surface_name : str
        Folder name of the nearest-matching template.
    surface_path : str
        Path to the templates directory.
    """

    if (
        kwargs.get("SL_source_space") == "volumetric"
        and kwargs.get("parcellation_parc") == "aparc.a2009s"
    ):
        produce_aparc_a2009s_aseg(
            save_path=kwargs.get("freesurfer_template_path"),
            freesurfer_home=kwargs.get("freesurfer_home"),
            freesurfer_license=kwargs.get("freesurfer_license"),
        )

    temp_path = os.path.join(project_dir, "Configurations", "runner_params.json")
    with open(temp_path, "r") as file:
        runner_params = json.load(file)

    dataset_name = runner_params["subjects"][subject]["dataset_name"]
    demographic_file_p = os.path.join(
        runner_params["datasets"][dataset_name]["base_dir"], "participants_bids.tsv"
    )
    demographic_file = pd.read_csv(demographic_file_p, sep="\t", index_col=0)
    demographic_file.index = demographic_file.index.astype(str)
    age = demographic_file.loc[subject]["age"]

    age_months = age * 12
    surface_name, surface_path = nearest_template_dir(
        age_months=age_months, subjects_dir=kwargs.get("freesurfer_template_path")
    )

    return surface_name, surface_path
