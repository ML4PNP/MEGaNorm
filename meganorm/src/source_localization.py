import matplotlib.pyplot as plt
from pathlib import Path
from joblib import parallel_config, parallel_backend
import subprocess
import numpy as np
import logging
import shutil
import mne
import joblib
import os

from meganorm.src.preprocess import check_tsss

logger = logging.getLogger(__name__)

def run_recon_freesurfer(
        freesurfer_home: str,
        subjects_dir: str,
        license_path: str,
        subject_id: str,
        mri_path: str
):
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
    result = subprocess.run(["which", "recon-all"], capture_output=True, text=True, env=env)
    recon_path = result.stdout.strip()

    if recon_path:
        freesurfer_home = os.path.abspath(os.path.join(os.path.dirname(recon_path), ".."))
    else:
        # Try common install paths
        possible_paths = [
            "/opt/freesurfer",
            os.path.expanduser("~/software/freesurfer"),
            os.path.expanduser("~/freesurfer"),
        ]
        freesurfer_home = next((p for p in possible_paths if os.path.exists(os.path.join(p, "SetUpFreeSurfer.sh"))), None)

    if not freesurfer_home:
        error_msg = "FreeSurfer not found. Please install it (https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall)" \
        " or set FREESURFER_HOME manually if it has already been installed."
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
    
    os.makedirs(
        os.path.join(
            figures_path,
            "QC_results"
        ),
        exist_ok=True
    )

    # This lines of codes were copied from MNE:
    # https://mne.tools/stable/index.html
    # ====================================================================
    info, C, ch_names, idx_names = mne.viz.misc._index_info_cov(info, data_cov, exclude=exclude)

    for k, (idx, name, unit, scaling, key) in enumerate(idx_names):
        this_C = C[idx][:, idx]
        s = np.linalg.svd(this_C, compute_uv=False)

        this_C = mne.cov.Covariance(this_C, [info["ch_names"][ii] for ii in idx], [], [], 0)
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
        logger.info(f"The largest drop (cliff) in the singular value spectrum: {eigen_idx}")
        if not this_rank > eigen_idx:

            if subject in qc_ignore:
                pass

            else:

                _, fig = data_cov.plot(info=info)
                fig.savefig(
                    os.path.join(
                        figures_path,
                        "QC_results",
                        f"{subject}_QC_rank.png"
                    )
                )

                error_msg = f"There seems to be a problem with the covariance matrix for {name}. The estimated "\
                                "rank is higher than the largest drop (cliff) in the singular value spectrum. "\
                                "The spectrum for this subject will be saved—please review it. If you determine "\
                                "there is no issue, you can add the subject ID to the qc_ignore argument."
                logger.error(error_msg)
                raise Exception(error_msg)

    return 


def corregistration(data, 
                subject,
                subjects_dir, 
                plot_3d,
                **kwargs):
    """
    Coregister MEG data to MRI using fiducial alignment and iterative closest point (ICP).

    This function aligns the MEG head coordinate system to the MRI coordinate system using:
    1. Fiducial point fitting (nasion and preauricular points).
    2. Iterative Closest Point (ICP) alignment using head shape points.
    3. Automatic removal of poorly fitting head shape points based on a distance threshold.
    4. A final ICP refinement with adjusted parameters.

    Optionally, a 3D plot can be displayed to assess the quality of coregistration.

    Parameters
    ----------
    data : mne.io.Raw
        Raw MEG data with digitized head shape and fiducial points.
    subject : str
        Subject name (must match an entry in the `subjects_dir`).
    subjects_dir : str
        Path to the FreeSurfer `SUBJECTS_DIR` containing MRI reconstructions.
    plot_3d : bool
        If True, display a 3D visualization of the final coregistration.
    **kwargs : dict, optional
        Optional parameters to control the coregistration steps:

    -    fiducials : str default="estimated"
            Fiducial information. Can be `'auto'`, `'estimated'`, or a dict/list of points.
        - coregisteration_initial_n_iterations : int, default=6
            Number of ICP iterations for the initial fit.
        - coregisteration_initial_nasion_weight : float, default=2.0
            Weight given to the nasion during the initial ICP step.
        - coregisteration_distance_thr : float, default=0.005
            Distance threshold in meters for omitting poorly fitting head shape points.
        - coregisteration_final_n_iterations : int, default=20
            Number of ICP iterations for the final fit.
        - coregisteration_final_nasion_weight : float, default=10.0
            Nasion weight used during the final ICP step.

    References
    ----------
    https://mne.tools/stable/auto_tutorials/forward/25_automated_coreg.html

    Returns
    -------
    None
        The function modifies and visualizes the internal alignment state but does not return any object.
    """

    # creates a standard head model subject sample (required for coregisteration)
    if not os.path.exists(
        os.path.join(
            subjects_dir,
            subject,
            "bem",
            "inner_skull.surf"
        )
    ):
        
        logger.info("bem surface was not found; Creating a bem surface for the subject")

        mne.bem.make_watershed_bem(
            subject=subject,
            subjects_dir=subjects_dir,
            overwrite=True,
            gcaatlas=kwargs.get("gcaatlas", True),
            volume="T1", # TODO: this should be a data specific info
            preflood=kwargs.get("preflood", None)
        )
    
    coreg = mne.coreg.Coregistration(data.info, 
                            subject=subject,
                            subjects_dir=subjects_dir, 
                            fiducials=kwargs.get("corregistration_fiducials", "estimated"))
    
    # initial fit with fudicials
    coreg.fit_fiducials()

    # fit with icp
    coreg.fit_icp(n_iterations=kwargs.get("coregisteration_initial_n_iterations", 6),
            nasion_weight=kwargs.get("coregisteration_initial_nasion_weight", 2.0),
            verbose=True)

    # removing bad head shape points that were not fitted on the brain nicely
    # Removing any head shape point that is more than 5 mm away from the fitted MRI surface.
    coreg.omit_head_shape_points(distance=kwargs.get("coregisteration_distance_thr", 5.0/1000)) # distance is in meter

    coreg.fit_icp(n_iterations=kwargs.get("coregisteration_final_n_iterations", 20), 
                nasion_weight=kwargs.get("coregisteration_final_nasion_weight", 10.0), 
                verbose=True)

    if plot_3d:
        plot_kwargs = dict(
            subject=subject,
            subjects_dir=subjects_dir,
            surfaces="head",
            dig=True,
            eeg=[],
            meg="sensors",
            show_axes=True,
            coord_frame="meg",
        )
        fig = mne.viz.plot_alignment(data.info, trans=coreg.trans, **plot_kwargs)

    logger.info("Automatic coregisteration is done!")

    return coreg


def forward_solution(
        subject,
        subjects_dir,
        data,
        transformation_matrix,
        conductivity,
        source_space,
        **kwargs
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
    
    # source space
    if source_space == "surface":
        src = mne.setup_source_space(
                subject=subject,
                subjects_dir=subjects_dir,
                spacing=kwargs.get("source_space_spacing", "ico6"),
                add_dist=kwargs.get("source_space_add_dist", "patch"),
                n_jobs=kwargs.get("n_jobs", -1)
        )

    elif source_space == "volumetric":
        src = mne.setup_volume_source_space(
                subject=subject,
                subjects_dir=subjects_dir,
                surface= Path(subjects_dir) / subject / "bem" / "inner_skull.surf",
                add_interpolator=True,
                n_jobs=kwargs.get("n_jobs", -1)
        )

    # forward model
    model = mne.make_bem_model(
        subject=subject,
        ico=kwargs.get("source_space_spacing_ico", 6),
        conductivity=conductivity,
        subjects_dir=subjects_dir
    )

    bem = mne.make_bem_solution(model)
    logger.info(f"{source_space} BEM model with {len(conductivity)} layer/s was constructed.")

    lead_field_matrix = mne.make_forward_solution(
        data.info,
        trans=transformation_matrix,
        src=src,
        bem=bem,
        meg=kwargs.get("meg", True),
        eeg=kwargs.get("eeg", False),
        mindist=kwargs.get("forward_mindist", 5.0),
        n_jobs=kwargs.get("n_jobs", -1),
        verbose=True,
        ignore_ref=kwargs.get("source_localization_ignore_ref", True)
    )
    logger.info("Lead field matrix was estimated.")

    return lead_field_matrix, lead_field_matrix["src"]

def inverse_solution(
        subject,
        data,
        fwd,
        inverse_operator,
        figures_path,
        which_sensor,
        empty_room_recording=None,
        qc_ignore=[],
        number_of_reduced_ic=0,
        **kwargs
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
        data_rank = mne.compute_rank(data, rank="info")
        # since ICA has also removed some components
        data_rank[list(data_rank.keys())[0]] -= number_of_reduced_ic # TODO: not accuarte
    else:
        data_rank = mne.compute_rank(data)
    
    if empty_room_recording:
        noise_cov = mne.compute_raw_covariance(
            empty_room_recording,
            method=kwargs.get("covariance_method", "empirical"),
            n_jobs=kwargs.get("n_jobs", -1)
        ) # TODO: change to epoch later
        logger.info("Noise covariance was calculated from  empty room recordings. This will be used to pre-whiten" \
                    "the data")
        # If empty room recording is available, the rank for
        # lcmv should be the ranke of this recording
        lcmv_rank = mne.compute_rank(empty_room_recording)
        
        if data_rank["mag"] < lcmv_rank["mag"]:
            lcmv_rank["mag"] = data_rank["mag"].copy()
        if data_rank["grad"] < lcmv_rank["grad"]:
            lcmv_rank["grad"] = data_rank["grad"].copy()
    
    elif which_sensor["meg"]:
        # If both MAG and Grad are present, having a noise cov is necessary
        # to scale the data. If empty room recording is not available,
        # make_ad_hoc_cov make a diagonal covariance matrix where the 
        # diagonals represent channel wise variance and off-diagonals are zero.
        # The default noise values are 5 fT/cm, 20 fT for gradiometers, magnetometers
        noise_cov = mne.make_ad_hoc_cov(info=data.info,
                                        std=kwargs.get("ad_hoc_cov_std", None))
        lcmv_rank = data_rank.copy()

    else:
        logger.warning("Noise covariance is not calculated due to missing empty room recordings. Noise covariance can be" \
        "benefitial for prewhitenning activities across sensors with different scale and noise." \
        "Consider using noise covariance for a better results.")
        noise_cov=None
        # If empty room recording is not available, the rank for
        # lcmv should be the ranke of the data itself
        lcmv_rank = data_rank.copy()

    if inverse_operator == "lcmv":
        logger.info(f"Solving the inverse problem using {inverse_operator} algorithm. "\
                    f"A regularization of {kwargs.get('inverse_regularization_value', 0.05)} will be used " \
                    f"to shift the matrix so it can be invertible. Furthermore, we will use {kwargs.get('beamformer_pick_ori', "max-power")} for `pick_ori`.")
        # compute data covaraince
        data_cov = mne.compute_raw_covariance(
            data,
            method=kwargs.get("covariance_method", "empirical"),
            rank=data_rank,
            n_jobs=kwargs.get("n_jobs", -1)
        )
        
        if (kwargs.get("beamformer_pick_ori", "max_power") == "vector" and
            kwargs.get("beamformer_weight_norm", "unit-noise-gain") != "unit-noise-gain-invariant"):
            error_msg = "If you wish to compute a vector beamformer, it is necessary to use" \
                        " unit-noise-gain-invariant for weight_norm argument."
            logger.error(error_msg)
            raise Exception(error_msg)

        rank_based_quality_control(
            data_cov=data_cov,
            info=data.info,
            subject=subject,
            figures_path=figures_path,
            exclude=[], #TODO
            qc_ignore=qc_ignore)
        
        filters = mne.beamformer.make_lcmv(
            data.info,
            forward=fwd,
            data_cov=data_cov,
            noise_cov=noise_cov,
            reg=kwargs.get("inverse_regularization_value", 0.05), # for regularization (shifting the matrix)
            pick_ori=kwargs.get("beamformer_pick_ori", "max-power"),
            weight_norm=kwargs.get("beamformer_weight_norm", "unit-noise-gain"),
            # TODO: if rank==None, it will compute the rank
            # If rank==info, it will read from the info
            rank=lcmv_rank,
        )

    stc = mne.beamformer.apply_lcmv_raw(
        data,
        filters=filters
    )

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
        **kwargs
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

    stc


    src_from : instance of mne.SourceSpaces or mne.SourceEstimate


        # TODO
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
    morph : mne.SourceMorph
        The morph object used to perform the morphing, which can be reused or saved.

    Notes
    -----
    - The function currently uses a fixed `spacing=5`, but this can be overridden via `**kwargs`.
    - The `src` argument is included for completeness but not directly used.
    - Only surface source estimates are supported.
    """
    logger.info("Morphing the estimated source data onto a common space")

    if source_space == "surface":
        src_morph_to = mne.setup_source_space(
            subject=subject_to,
            subjects_dir=subjects_dir,
            spacing=kwargs.get("source_space_spacing", "ico6"),
            add_dist=kwargs.get("source_space_add_dist", "patch"),
            n_jobs=kwargs.get("n_jobs", -1)
        )
    elif source_space == "volumetric": # TODO

        inner_skull_path = Path(subjects_dir) / subject_to / "bem" / "inner_skull.surf"
        if not os.path.exists(inner_skull_path):
            mne.bem.make_watershed_bem(
                subject=subject_to,
                subjects_dir=subjects_dir,
                overwrite=True,
                gcaatlas=kwargs.get("gcaatlas", True),
                volume="T1", # TODO: this should be a data specific info.
                preflood=kwargs.get("preflood", None)
            )

        src_morph_to = mne.setup_volume_source_space(
                subject=subject_to,
                subjects_dir=subjects_dir,
                surface= inner_skull_path,
                add_interpolator=True,
                n_jobs=kwargs.get("n_jobs", -1)
        )

    logger.info("hello")
    with parallel_backend("threading"):
        with parallel_config(n_jobs=1):
            morph = mne.compute_source_morph(src_from, 
                                        subject_from=subject, 
                                        subject_to=subject_to, 
                                        subjects_dir=subjects_dir, 
                                        spacing=kwargs.get("spacing", 5), # TODO: this 5 should another spacing mentioned above
                                        src_to=src_morph_to
                                        )
        
            logger.info("hello again")
            stc_fsaverage = morph.apply(stc)
            logger.info("bye")

    if plot_3d:
        brain = stc_fsaverage.plot(
            subjects_dir=subjects_dir,
            clim=dict(kind="value", lims=[3, 6, 9]),
            smoothing_steps=7,
        )

    logger.info("Morphing is finised!")
    return stc_fsaverage, src_morph_to


def parcellate(
        subject,
        subjects_dir,
        stc_fsaverage,
        src_morph,
        source_space,
        **kwargs
    ):
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
    stc_fsaverage : mne.SourceEstimate | list of SourceEstimate
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
    logger.info(f"Parcellating morphed source estimates using {kwargs.get('parcellation_parc', 'aparc.a2009s')} atlas.")

    if not os.path.exists(
        os.path.join(
            subjects_dir,
            subject
        )):
        mne.datasets.fetch_fsaverage()

    if source_space == "surface":
        labels = mne.read_labels_from_annot(
            subject=subject,
            subjects_dir=subjects_dir,
            parc=kwargs.get("parcellation_parc", "aparc.a2009s")
        )
        labels_list = [label.name for label in labels]
    elif source_space == "volumetric":
        parc = kwargs.get("parcellation_parc", "aparc.a2009s")
        labels = os.path.join(subjects_dir, subject, "mri", f"{parc}+aseg.mgz")
        labels_list = mne.get_volume_labels_from_aseg(mgz_fname=labels, return_colors=False)

    else:
        error_msg = "Source space model is not detected. Source splace must be either 'surface' or 'volumetric'."
        logger.ERROR(error_msg)
        raise ValueError(error_msg)

    parcelled_stc = mne.extract_label_time_course(
        stcs=stc_fsaverage,
        labels=labels,
        src=src_morph,
        mode=kwargs.get("parcellation_mode", "mean"),
        return_generator=False
    )

    logger.info("Parcellation is finised!")

    return parcelled_stc, labels_list


def source_localization(
        subject,
        subjects_dir,
        subject_to,
        data,
        figures_path,
        which_sensor,
        source_space="surface",
        conductivity=(0.3,),
        inverse_operator="lcmv",
        plot_3d=False,
        qc_ignore=[],
        empty_room_recording=None,
        number_of_reduced_ic=0,
        **kwargs
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
    empty_room_recording: TODO
    
    plot_3d:TODO

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

    coreg = corregistration(
        data=data, 
        subject=subject,
        subjects_dir=subjects_dir, 
        plot_3d=plot_3d,
        **kwargs
    )

    fwd, src = forward_solution(
        subject=subject,
        subjects_dir=subjects_dir,
        data=data,
        transformation_matrix=coreg.trans,
        conductivity=conductivity,
        source_space=source_space,
        **kwargs
    )

    del coreg

    stc = inverse_solution(
        subject=subject,
        data=data,
        fwd=fwd,
        inverse_operator=inverse_operator,
        empty_room_recording=empty_room_recording,
        figures_path=figures_path,
        qc_ignore=qc_ignore,
        number_of_reduced_ic=number_of_reduced_ic,
        which_sensor=which_sensor,
        **kwargs
    )

    del fwd

    stc_fsaverage, src_morph = morph_stc(
        subject=subject,
        subject_to=subject_to,
        subjects_dir=subjects_dir,
        stc=stc,
        src_from=src,
        source_space=source_space,
        plot_3d=plot_3d,
        **kwargs
        )
    
    del stc
    del src

    stc, labels = parcellate(
            subject=subject_to,
            subjects_dir=subjects_dir,
            stc_fsaverage=stc_fsaverage,
            src_morph=src_morph,
            source_space=source_space,
            **kwargs
    )
    del stc_fsaverage
    del src_morph

    logger.info("Done; congrats! ")

    return stc, labels

def numpy_to_mne_raw(stc, labels, ch_name, sampling_rate):
    """
    Convert a parcellated source estimate into an MNE Raw object.

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
    >>> raw = numpy_to_mne_raw(parcelled_stc, labels, ch_name='misc', sampling_rate=1000)
    >>> raw.plot()
    """
    ch_types = [ch_name] * len(labels)
    info = mne.create_info(ch_names=labels, sfreq=sampling_rate, ch_types=ch_types)
    raw_parc = mne.io.RawArray(stc, info)
    return raw_parc
