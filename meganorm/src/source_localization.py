import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import mne
import os


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

    # Source FreeSurfer setup script
    setup_command = f"source {freesurfer_home}/SetUpFreeSurfer.sh"

    # Construct recon-all command
    recon_command = f"recon-all -i {mri_path} -s {subject_id} -all"

    # Combine into a full bash command
    full_command = f"""
    bash -c '
    {setup_command} &&
    # {recon_command}
    '
    """

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


    print("hi")


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
    try:
        mne.bem.make_watershed_bem(
            subject=subject,
            subjects_dir=subjects_dir,
            overwrite=False
        )
    except RuntimeError as e:
        if "watershed already exists" in str(e):
            pass
    
    coreg = mne.coreg.Coregistration(data.info, 
                            subject=subject,
                            subjects_dir=subjects_dir, 
                            fiducials=kwargs.get("corregistration_fiducials", "estimated"))
    
    # initial fit with fudicials
    coreg.fit_fiducials(verbose=False)

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
        - source-space_add_dist : str, default="patch"
            Whether to add patch information to the surface source space.
        - forward_ico : int, default=5
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
                spacing=kwargs.get("source_space_spacing", "oct6"),
                add_dist=kwargs.get("source-space_add_dist", "patch")
        )

    elif source_space == "volumetric":
        src = mne.setup_volume_source_space(
                subject=subject,
                subjects_dir=subjects_dir,
                surface= Path(subjects_dir) / subject / "bem" / "inner_skull.surf",
                add_interpolator=True,
        )

    # forward model
    model = mne.make_bem_model(
        subject=subject,
        ico=kwargs.get("forward_ico", 5),
        conductivity=conductivity,
        subjects_dir=subjects_dir
    )

    bem = mne.make_bem_solution(model)

    lead_field_matrix = mne.make_forward_solution(
        data.info,
        trans=transformation_matrix,
        src=src,
        bem=bem,
        meg=kwargs.get("meg", True),
        eeg=kwargs.get("eeg", False),
        mindist=kwargs.get("forward_mindist", 5.0),
        n_jobs=kwargs.get("n_jobs", -1),
        verbose=True
    )

    return lead_field_matrix, src

def inverse_solution(
        data,
        fwd,
        inverse_operator,
        empty_room_recording_path=None,
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
    empty_room_recording_path : str or Path or None
        Path to empty-room MEG recording used to estimate noise covariance.
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
    
    if empty_room_recording_path:
        empty_room_rocord = mne.io.read_raw(empty_room_recording_path, preload=True)
        noise_cov = mne.compute_raw_covariance(
            empty_room_rocord,
            method=kwargs.get("covariance_method", "emperical")
        ) # TODO: change to epoch later
    else: 
        noise_cov=None

    if inverse_operator == "lcmv":

        # compute data covaraince
        data_cov = mne.compute_raw_covariance(
            data,
            method=kwargs.get("covariance_method", "empirical")
        )

        filters = mne.beamformer.make_lcmv(
            data.info,
            forward=fwd,
            data_cov=data_cov,
            noise_cov=noise_cov,
            reg=kwargs.get("inverse_regularization_value", 0.05), # for regularization (shifting the matrix)
            pick_ori=kwargs.get("beamformer_pick_ori", "max-power"),
            weight_norm=kwargs.get("beamformer_weight_norm", "unit-noise-gain"),
            rank=None,
        )

    stc = mne.beamformer.apply_lcmv_raw(
        data,
        filters=filters
    )

    return stc


def morph_stc(
        subject,
        subject_to,
        subjects_dir,
        src,
        stc,
        plot_gui=False,
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
    src : instance of mne.SourceSpaces
        Source space of the original subject. (Note: currently unused in the function.)
    stc : instance of mne.SourceEstimate
        The source estimate object to morph.
    plot_gui : bool, optional
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
    # src[0]['subject_his_id'] = ""
    # src[1]['subject_his_id'] = "" TODO
    morph = mne.compute_source_morph(stc, 
                                    subject_from=subject, 
                                    subject_to=subject_to, 
                                    subjects_dir=subjects_dir, 
                                    spacing=kwargs.get("spacing", 5),
                                    # src_to=src # TODO
                                    )
    
    stc_fsaverage = morph.apply(stc)

    if plot_gui:
        brain = stc_fsaverage.plot(
            subjects_dir=subjects_dir,
            clim=dict(kind="value", lims=[3, 6, 9]),
            smoothing_steps=7,
        )

    return stc_fsaverage, morph


def parcellate(
        subject,
        subjects_dir,
        stc_fsaverage,
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
    
    labels = mne.read_labels_from_annot(
        subject=subject,
        subjects_dir=subjects_dir,
        parc=kwargs.get("parcellation_parc", "aparc.a2009s")
    )

    if source_space == "surface":
        src_morph = mne.setup_source_space(
            subject=subject,
            subjects_dir=subjects_dir,
            spacing=kwargs.get("source_space_spacing", "ico6"),
            add_dist=kwargs.get("source_space_add_dist", "patch"),
        )
    elif source_space == "volumetric":
        # src = mne.setup_volume_source_space(
        #         subject=subject,
        #         subjects_dir=subjects_dir,
        #         surface= Path(subjects_dir) / subject / "bem" / "inner_skull.surf",
        #         add_interpolator=True,
        # )
        pass

    parcelled_stc = mne.extract_label_time_course(
        stcs=stc_fsaverage,
        labels=labels,
        src=src_morph,
        mode=kwargs.get("parcellation_mode", "mean_flip"),
        return_generator=False
    )

    return parcelled_stc


def source_localization(
        subject,
        subjects_dir,
        subject_to,
        data,
        freesurfer_path,
        freesurfer_license_path,
        source_space="surface",
        conductivity=(0.3,),
        inverse_operator="lcmv",
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
    **kwargs : dict
        Additional keyword arguments passed to internal functions, such as:
        
        - parcellation_parc : str
            Parcellation scheme used in `parcellate()` (e.g., 'aparc', 'aparc.a2009s').
        - source_space_spacing : str | int
            Spacing resolution for the source space setup (e.g., 'ico5', 'oct6').
        - parcellation_mode : str
            Method to summarize activity within each label (e.g., 'mean', 'mean_flip').
        - plot_gui : bool
            Whether to visualize the morphed source estimate.

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

    set_freesurfer_paths(
        freesurfer_home=freesurfer_path,
        subjects_dir=subjects_dir,
        license_path=freesurfer_license_path,
    )

    coreg = corregistration(
        data=data, 
        subject=subject,
        subjects_dir=subjects_dir, 
        plot_3d=False
    )

    fwd, src = forward_solution(
        subject=subject,
        subjects_dir=subjects_dir,
        data=data,
        transformation_matrix=coreg.trans,
        conductivity=conductivity,
        source_space=source_space
    )

    stc = inverse_solution(
        data=data,
        fwd=fwd,
        inverse_operator=inverse_operator
    )

    stc_fsaveage, _ = morph_stc(
        subject=subject,
        subject_to=subject_to,
        subjects_dir=subjects_dir,
        src=src,
        stc=stc,
        plot_gui=False,
        )
    
    stc = parcellate(
            subject=subject_to,
            subjects_dir=subjects_dir,
            stc_fsaverage=stc_fsaveage,
            source_space=source_space
    )

    return stc