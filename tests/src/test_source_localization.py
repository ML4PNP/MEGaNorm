import logging
import os
import subprocess
from types import SimpleNamespace

import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest
from mne.io.constants import FIFF
from meganorm.src import source_localization as sl

from meganorm.src.source_localization import (
    build_template_index,
    capture_mne_log,
    check_digitization_points,
    check_freesurfer,
    check_tsss,
    corregistration,
    forward_solution,
    inverse_solution,
    make_bem_model,
    morph_stc,
    nearest_template_dir,
    numpy_to_mne_epoch,
    parcellate,
    prepare_template,
    produce_aparc_a2009s_aseg,
    regularized_cov_condition,
    run_recon_freesurfer,
    save_bem_figure,
    save_cov_figures,
    set_freesurfer_paths,
)


def make_fake_freesurfer(tmp_path, exit_code=0):
    freesurfer_home = tmp_path / "FreeSurfer Home"
    freesurfer_bin = freesurfer_home / "bin"
    freesurfer_bin.mkdir(parents=True)
    setup_script = freesurfer_home / "SetUpFreeSurfer.sh"
    setup_script.write_text(
        'export SETUP_MARKER="loaded"\n'
        'export PATH="$FREESURFER_HOME/bin:$PATH"\n'
        'export FS_LICENSE="/setup/override-license.txt"\n'
        'export FREESURFER_LICENSE="/setup/override-legacy-license.txt"\n'
    )
    recon_all = freesurfer_bin / "recon-all"
    recon_all.write_text(
        "#!/bin/sh\n"
        "{\n"
        "  printf 'arg=%s\\n' \"$@\"\n"
        "  printf 'FREESURFER_HOME=%s\\n' \"$FREESURFER_HOME\"\n"
        "  printf 'SUBJECTS_DIR=%s\\n' \"$SUBJECTS_DIR\"\n"
        "  printf 'FS_LICENSE=%s\\n' \"$FS_LICENSE\"\n"
        "  printf 'FREESURFER_LICENSE=%s\\n' \"$FREESURFER_LICENSE\"\n"
        "  printf 'SETUP_MARKER=%s\\n' \"$SETUP_MARKER\"\n"
        "  printf 'PATH=%s\\n' \"$PATH\"\n"
        '} > "$RECON_RECORD"\n'
        f"exit {exit_code}\n"
    )
    recon_all.chmod(0o755)
    return freesurfer_home


class FakeCoregistration:
    def __init__(self, info, subject, subjects_dir, fiducials):
        self.init_args = {
            "info": info,
            "subject": subject,
            "subjects_dir": subjects_dir,
            "fiducials": fiducials,
        }
        self.scale = np.array([1.1, 1.1, 1.1])
        self.trans = "head-to-mri"
        self.calls = []

    def set_scale_mode(self, mode):
        self.calls.append(("set_scale_mode", mode))

    def fit_fiducials(self):
        self.calls.append(("fit_fiducials",))

    def fit_icp(self, **kwargs):
        self.calls.append(("fit_icp", kwargs))

    def omit_head_shape_points(self, distance):
        self.calls.append(("omit_head_shape_points", distance))

    def compute_dig_mri_distances(self):
        self.calls.append(("compute_dig_mri_distances",))
        return np.array([0.001, 0.003])


@pytest.mark.unit
def test_corregistration_runs_two_stage_icp_when_head_shape_points_exist(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(mne.coreg, "Coregistration", FakeCoregistration)
    monkeypatch.setattr(
        sl, "check_digitization_points", lambda data, logger: (3, 0, 0, 0)
    )
    data = SimpleNamespace(info="measurement-info")

    coreg, fit_subject, returned_subjects_dir = corregistration(
        data=data,
        subject="fsaverage",
        subjects_dir=tmp_path,
        participant_id="sub-01",
        coregisteration_initial_n_iterations=4,
        coregisteration_initial_nasion_weight=1.5,
        coregisteration_distance_thr=0.004,
        coregisteration_final_n_iterations=12,
        coregisteration_final_nasion_weight=8.0,
    )

    assert fit_subject == "fsaverage"
    assert returned_subjects_dir == tmp_path
    assert coreg.init_args == {
        "info": "measurement-info",
        "subject": "fsaverage",
        "subjects_dir": tmp_path,
        "fiducials": "estimated",
    }
    assert coreg.calls == [
        ("fit_fiducials",),
        ("fit_icp", {"n_iterations": 4, "nasion_weight": 1.5, "verbose": True}),
        ("omit_head_shape_points", 0.004),
        ("fit_icp", {"n_iterations": 12, "nasion_weight": 8.0, "verbose": True}),
        ("compute_dig_mri_distances",),
    ]


@pytest.mark.unit
def test_corregistration_scales_template_for_participant(monkeypatch, tmp_path):
    scale_calls = []
    monkeypatch.setattr(mne.coreg, "Coregistration", FakeCoregistration)
    monkeypatch.setattr(
        sl, "check_digitization_points", lambda data, logger: (0, 0, 0, 0)
    )
    monkeypatch.setattr(mne, "scale_mri", lambda **kwargs: scale_calls.append(kwargs))

    coreg, fit_subject, returned_subjects_dir = corregistration(
        data=SimpleNamespace(info="measurement-info"),
        subject="template-18",
        subjects_dir=tmp_path,
        participant_id="sub-02",
        apply_mri_template=True,
        coregisteration_scale_mode="uniform",
    )

    assert fit_subject == "sub-02_scaled"
    assert returned_subjects_dir == tmp_path
    assert coreg.calls == [("set_scale_mode", "uniform"), ("fit_fiducials",)]
    assert len(scale_calls) == 1
    assert scale_calls[0] == {
        "subject_from": "template-18",
        "subject_to": "sub-02_scaled",
        "scale": coreg.scale,
        "subjects_dir": tmp_path,
        "overwrite": True,
        "labels": True,
        "annot": True,
        "skip_fiducials": True,
    }


@pytest.mark.unit
def test_morph_stc_builds_surface_target_and_applies_morph(monkeypatch, tmp_path):
    calls = {}

    class FakeMorph:
        def apply(self, stc):
            calls["applied_to"] = stc
            return "morphed-stc"

    monkeypatch.setattr(
        mne,
        "setup_source_space",
        lambda **kwargs: calls.setdefault("target_source_args", kwargs)
        and "target-source",
    )

    def fake_compute_source_morph(src_from, **kwargs):
        calls["morph"] = {"src_from": src_from, **kwargs}
        return FakeMorph()

    monkeypatch.setattr(mne, "compute_source_morph", fake_compute_source_morph)

    result, target_source = morph_stc(
        subject="sub-01",
        subject_to="fsaverage",
        subjects_dir=tmp_path,
        stc="subject-stc",
        src_from="subject-source",
        source_space="surface",
        source_space_spacing="oct5",
        source_space_add_dist=False,
        source_space_spacing_number=5,
        n_jobs=2,
    )

    assert result == "morphed-stc"
    assert target_source == "target-source"
    assert calls["target_source_args"] == {
        "subject": "fsaverage",
        "subjects_dir": tmp_path,
        "spacing": "oct5",
        "add_dist": False,
        "n_jobs": 2,
    }
    assert calls["morph"] == {
        "src_from": "subject-source",
        "subject_from": "sub-01",
        "src_to": "target-source",
        "subject_to": "fsaverage",
        "subjects_dir": tmp_path,
        "spacing": 5,
    }
    assert calls["applied_to"] == "subject-stc"


@pytest.mark.unit
def test_morph_stc_applies_morph_to_each_epoch(monkeypatch, tmp_path):
    applied = []

    class FakeMorph:
        def apply(self, stc):
            if isinstance(stc, list):
                raise TypeError("SourceMorph.apply accepts one source estimate")
            applied.append(stc)
            return f"morphed-{stc}"

    monkeypatch.setattr(mne, "setup_source_space", lambda **kwargs: "target-source")
    monkeypatch.setattr(
        mne, "compute_source_morph", lambda *args, **kwargs: FakeMorph()
    )

    result, target_source = morph_stc(
        subject="sub-01",
        subject_to="fsaverage",
        subjects_dir=tmp_path,
        stc=["epoch-1", "epoch-2"],
        src_from="subject-source",
        source_space="surface",
    )

    assert result == ["morphed-epoch-1", "morphed-epoch-2"]
    assert target_source == "target-source"
    assert applied == ["epoch-1", "epoch-2"]


@pytest.mark.unit
def test_morph_stc_requires_single_estimate_for_3d_plot(monkeypatch, tmp_path):
    class FakeMorph:
        def apply(self, stc):
            return object()

    monkeypatch.setattr(mne, "setup_source_space", lambda **kwargs: "target-source")
    monkeypatch.setattr(
        mne, "compute_source_morph", lambda *args, **kwargs: FakeMorph()
    )

    with pytest.raises(ValueError, match="plot_3d.*single source estimate"):
        morph_stc(
            subject="sub-01",
            subject_to="fsaverage",
            subjects_dir=tmp_path,
            stc=["epoch-1", "epoch-2"],
            src_from="subject-source",
            source_space="surface",
            plot_3d=True,
        )


@pytest.mark.unit
def test_morph_stc_builds_missing_volumetric_target(monkeypatch, tmp_path):
    calls = {}
    inner_skull = tmp_path / "fsaverage" / "bem" / "inner_skull.surf"

    class FakeMorph:
        def apply(self, stc):
            return "morphed-volume-stc"

    monkeypatch.setattr(
        mne.bem,
        "make_watershed_bem",
        lambda **kwargs: calls.setdefault("watershed", kwargs),
    )
    monkeypatch.setattr(
        mne,
        "setup_volume_source_space",
        lambda **kwargs: calls.setdefault("target_source", kwargs)
        and "target-volume-source",
    )
    monkeypatch.setattr(
        mne, "compute_source_morph", lambda *args, **kwargs: FakeMorph()
    )

    result, target_source = morph_stc(
        subject="sub-01",
        subject_to="fsaverage",
        subjects_dir=tmp_path,
        stc="subject-stc",
        src_from="subject-source",
        source_space="volumetric",
        preflood=25,
    )

    assert result == "morphed-volume-stc"
    assert target_source == "target-volume-source"
    assert calls["watershed"] == {
        "subject": "fsaverage",
        "subjects_dir": tmp_path,
        "overwrite": True,
        "gcaatlas": True,
        "volume": "T1",
        "preflood": 25,
    }
    assert calls["target_source"]["surface"] == inner_skull


@pytest.mark.unit
def test_morph_stc_rejects_unsupported_source_space(tmp_path):
    with pytest.raises(ValueError, match="surface.*volumetric"):
        morph_stc(
            subject="sub-01",
            subject_to="fsaverage",
            subjects_dir=tmp_path,
            stc="subject-stc",
            src_from="subject-source",
            source_space="invalid",
        )


@pytest.mark.unit
def test_parcellate_extracts_surface_label_time_courses(monkeypatch, tmp_path):
    labels = [SimpleNamespace(name="left-region"), SimpleNamespace(name="right-region")]
    parcelled = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    calls = {}
    monkeypatch.setattr(mne, "read_labels_from_annot", lambda **kwargs: labels)

    def fake_extract_label_time_course(**kwargs):
        calls.update(kwargs)
        return parcelled

    monkeypatch.setattr(
        mne, "extract_label_time_course", fake_extract_label_time_course
    )

    result, names = parcellate(
        subject="fsaverage",
        subjects_dir=tmp_path,
        stc="morphed-stc",
        src="target-source",
        source_space="surface",
        parcellation_mode="mean_flip",
    )

    assert result is parcelled
    assert names == ["left-region", "right-region"]
    assert calls == {
        "stcs": "morphed-stc",
        "labels": labels,
        "src": "target-source",
        "mode": "mean_flip",
        "return_generator": False,
    }


@pytest.mark.unit
def test_parcellate_uses_volumetric_segmentation_labels(monkeypatch, tmp_path):
    calls = {}
    parcelled = np.array([[[1.0, 2.0]]])

    def fake_get_volume_labels(mgz_fname, return_colors):
        calls["labels"] = (mgz_fname, return_colors)
        return ["Left-Caudate"]

    def fake_extract_label_time_course(**kwargs):
        calls["extract"] = kwargs
        return parcelled

    monkeypatch.setattr(mne, "get_volume_labels_from_aseg", fake_get_volume_labels)
    monkeypatch.setattr(
        mne, "extract_label_time_course", fake_extract_label_time_course
    )

    result, names = parcellate(
        subject="fsaverage",
        subjects_dir=tmp_path,
        stc="morphed-stc",
        src="target-volume-source",
        source_space="volumetric",
        parcellation_parc="aparc.custom",
    )

    segmentation = str(tmp_path / "fsaverage" / "mri" / "aparc.custom+aseg.mgz")
    assert result is parcelled
    assert names == ["Left-Caudate"]
    assert calls["labels"] == (segmentation, False)
    assert calls["extract"]["labels"] == segmentation


@pytest.mark.unit
def test_parcellate_rejects_unsupported_source_space(tmp_path):
    with pytest.raises(ValueError, match="surface.*volumetric"):
        parcellate(
            subject="fsaverage",
            subjects_dir=tmp_path,
            stc="morphed-stc",
            src="target-source",
            source_space="invalid",
        )


@pytest.mark.unit
def test_inverse_solution_uses_info_rank_and_ad_hoc_noise_covariance(
    monkeypatch, tmp_path
):
    data = SimpleNamespace(info="data-info")
    segments = SimpleNamespace(info="segments-info")
    noise_cov = object()
    data_cov = object()
    filters = object()
    calls = {}

    monkeypatch.setattr(sl, "check_tsss", lambda meg_data: True)

    def fake_compute_rank(instance, rank=None):
        calls["rank"] = (instance, rank)
        return {"mag": 4}

    monkeypatch.setattr(mne, "compute_rank", fake_compute_rank)
    monkeypatch.setattr(
        mne,
        "make_ad_hoc_cov",
        lambda info, std: calls.setdefault("ad_hoc", (info, std)) and noise_cov,
    )
    monkeypatch.setattr(
        mne,
        "compute_raw_covariance",
        lambda instance, **kwargs: calls.setdefault("data_cov", (instance, kwargs))
        and data_cov,
    )
    monkeypatch.setattr(sl, "save_cov_figures", lambda *args, **kwargs: None)

    def fake_make_lcmv(info, **kwargs):
        calls["lcmv"] = {"info": info, **kwargs}
        return filters

    monkeypatch.setattr(mne.beamformer, "make_lcmv", fake_make_lcmv)
    monkeypatch.setattr(
        mne.beamformer,
        "apply_lcmv_epochs",
        lambda epochs, filters: ("source-estimate", epochs, filters),
    )

    result = inverse_solution(
        subject="sub-01",
        data=data,
        segments=segments,
        fwd="forward-model",
        inverse_operator="lcmv",
        project_dir=tmp_path,
        which_sensor_dict={"mag": True},
        source_space="surface",
        ad_hoc_cov_std={"mag": 1e-14},
        inverse_regularization_value=0.1,
        beamformer_pick_ori="normal",
        beamformer_weight_norm="nai",
        n_jobs=2,
    )

    assert result == ("source-estimate", segments, filters)
    assert calls["rank"] == (data, "info")
    assert calls["ad_hoc"] == ("data-info", {"mag": 1e-14})
    assert calls["lcmv"] == {
        "info": "segments-info",
        "forward": "forward-model",
        "data_cov": data_cov,
        "noise_cov": noise_cov,
        "reg": 0.1,
        "pick_ori": "normal",
        "weight_norm": "nai",
        "rank": {"mag": 4},
        "depth": None,
    }


@pytest.mark.unit
def test_inverse_solution_limits_empty_room_rank_to_data_rank(monkeypatch, tmp_path):
    data = SimpleNamespace(info="data-info")
    empty_room = SimpleNamespace(info="empty-room-info")
    segments = SimpleNamespace(info="segments-info")
    noise_cov = object()
    data_cov = object()
    calls = {"covariance": [], "figures": []}

    monkeypatch.setattr(sl, "check_tsss", lambda meg_data: False)

    def fake_compute_rank(instance, rank=None):
        if instance is data:
            calls["data_rank_argument"] = rank
            return {"mag": 3, "grad": 2}
        assert instance is empty_room
        return {"mag": 5, "grad": 1}

    def fake_compute_raw_covariance(instance, **kwargs):
        calls["covariance"].append((instance, kwargs))
        return noise_cov if instance is empty_room else data_cov

    monkeypatch.setattr(mne, "compute_rank", fake_compute_rank)
    monkeypatch.setattr(mne, "compute_raw_covariance", fake_compute_raw_covariance)
    monkeypatch.setattr(
        sl,
        "save_cov_figures",
        lambda cov, info, **kwargs: calls["figures"].append((cov, info, kwargs["tag"])),
    )
    monkeypatch.setattr(
        mne.beamformer,
        "make_lcmv",
        lambda info, **kwargs: calls.setdefault("lcmv", {"info": info, **kwargs}),
    )
    monkeypatch.setattr(
        mne.beamformer,
        "apply_lcmv_epochs",
        lambda epochs, filters: "source-estimates",
    )

    result = inverse_solution(
        subject="sub-02",
        data=data,
        segments=segments,
        fwd="forward-model",
        inverse_operator="lcmv",
        project_dir=tmp_path,
        which_sensor_dict={"mag": True, "grad": True},
        source_space="surface",
        empty_room_recording=empty_room,
    )

    assert result == "source-estimates"
    assert calls["data_rank_argument"] is None
    assert calls["lcmv"]["rank"] == {"mag": 3, "grad": 1}
    assert calls["lcmv"]["noise_cov"] is noise_cov
    assert calls["figures"] == [
        (noise_cov, "empty-room-info", "noiseCovariance"),
        (data_cov, "data-info", "dataCovariance"),
    ]


@pytest.mark.unit
def test_inverse_solution_requires_depth_for_volumetric_source(monkeypatch, tmp_path):
    data = SimpleNamespace(info="data-info")
    monkeypatch.setattr(
        sl,
        "check_tsss",
        lambda meg_data: pytest.fail("rank estimation must not start"),
    )

    with pytest.raises(ValueError, match="beamforme_depth"):
        inverse_solution(
            subject="sub-03",
            data=data,
            segments=SimpleNamespace(info="segments-info"),
            fwd="forward-model",
            inverse_operator="lcmv",
            project_dir=tmp_path,
            which_sensor_dict={"mag": True},
            source_space="volumetric",
        )


@pytest.mark.unit
def test_inverse_solution_rejects_unsupported_method_before_covariance(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        sl,
        "check_tsss",
        lambda meg_data: pytest.fail("rank estimation must not start"),
    )

    with pytest.raises(ValueError, match="only.*lcmv"):
        inverse_solution(
            subject="sub-04",
            data=SimpleNamespace(info="data-info"),
            segments=SimpleNamespace(info="segments-info"),
            fwd="forward-model",
            inverse_operator="dspm",
            project_dir=tmp_path,
            which_sensor_dict={"eeg": True},
            source_space="surface",
        )


@pytest.mark.unit
def test_source_localization_propagates_scaled_subject_through_morphing(
    monkeypatch, tmp_path
):
    subjects_dir = tmp_path / "subjects"
    scaled_subjects_dir = tmp_path / "scaled-subjects"
    inner_skull = subjects_dir / "sub-01" / "bem" / "inner_skull.surf"
    inner_skull.parent.mkdir(parents=True)
    inner_skull.touch()
    calls = {}
    parcelled = np.array([[[1.0, 2.0]]])

    def fake_corregistration(**kwargs):
        calls["coregistration"] = kwargs
        return (
            SimpleNamespace(trans="scaled-transform"),
            "sub-01_scaled",
            scaled_subjects_dir,
        )

    def fake_forward_solution(**kwargs):
        calls["forward"] = kwargs
        return "forward-model", "filtered-source"

    def fake_inverse_solution(**kwargs):
        calls["inverse"] = kwargs
        return ["epoch-1-stc", "epoch-2-stc"]

    def fake_morph_stc(**kwargs):
        calls["morph"] = kwargs
        return ["morphed-epoch-1", "morphed-epoch-2"], "target-source"

    def fake_parcellate(**kwargs):
        calls["parcellate"] = kwargs
        return parcelled, ["Left-Caudate"]

    monkeypatch.setattr(sl, "corregistration", fake_corregistration)
    monkeypatch.setattr(sl, "forward_solution", fake_forward_solution)
    monkeypatch.setattr(sl, "inverse_solution", fake_inverse_solution)
    monkeypatch.setattr(sl, "morph_stc", fake_morph_stc)
    monkeypatch.setattr(sl, "parcellate", fake_parcellate)

    result, labels = sl.source_localization(
        recording_path=tmp_path / "recording.fif",
        project_dir=tmp_path / "project",
        subject="sub-01",
        subjects_dir=subjects_dir,
        subject_to="fsaverage",
        data=SimpleNamespace(info={"dig": []}),
        segments="segments",
        figures_path=tmp_path / "figures",
        which_sensor_dict={"mag": True},
        source_space="surface",
        conductivity=(0.3,),
        apply_morphing=True,
        apply_mri_template=False,
        which_sensor="eeg",
        bem_plot_orientations=None,
    )

    assert result is parcelled
    assert labels == ["Left-Caudate"]
    assert calls["forward"]["subject"] == "sub-01_scaled"
    assert calls["forward"]["subjects_dir"] == scaled_subjects_dir
    assert calls["forward"]["transformation_matrix"] == "scaled-transform"
    assert calls["inverse"]["fwd"] == "forward-model"
    assert calls["morph"]["subject"] == "sub-01_scaled"
    assert calls["morph"]["src_from"] == "filtered-source"
    assert calls["morph"]["stc"] == ["epoch-1-stc", "epoch-2-stc"]
    assert calls["parcellate"]["subject"] == "fsaverage"
    assert calls["parcellate"]["src"] == "target-source"
    assert calls["parcellate"]["stc"] == ["morphed-epoch-1", "morphed-epoch-2"]


@pytest.mark.unit
@pytest.mark.parametrize("meg_key", ["meg", "grad", "mag"])
def test_forward_solution_configures_surface_model_and_returns_filtered_source(
    monkeypatch, tmp_path, meg_key
):
    calls = {}
    data = SimpleNamespace(info=object())
    forward = {"src": "filtered-source", "sol": "lead-field"}

    def fake_setup_source_space(**kwargs):
        calls["source"] = kwargs
        return "surface-source"

    def fake_make_bem_model(**kwargs):
        calls["bem_model"] = kwargs
        return "bem-model"

    def fake_make_bem_solution(model):
        calls["bem_solution"] = model
        return "bem-solution"

    def fake_make_forward_solution(info, **kwargs):
        calls["forward"] = {"info": info, **kwargs}
        return forward

    monkeypatch.setattr(mne, "setup_source_space", fake_setup_source_space)
    monkeypatch.setattr(mne, "make_bem_model", fake_make_bem_model)
    monkeypatch.setattr(mne, "make_bem_solution", fake_make_bem_solution)
    monkeypatch.setattr(mne, "make_forward_solution", fake_make_forward_solution)

    result, result_src = forward_solution(
        subject="sub-01",
        subjects_dir=tmp_path,
        data=data,
        transformation_matrix="head-to-mri",
        conductivity=(0.3,),
        source_space="surface",
        which_sensor_dict={meg_key: True, "eeg": False},
        source_space_spacing="oct5",
        source_space_add_dist=False,
        source_space_spacing_number=4,
        forward_mindist=3.0,
        source_localization_ignore_ref=False,
        n_jobs=2,
    )

    assert result is forward
    assert result_src == "filtered-source"
    assert calls == {
        "source": {
            "subject": "sub-01",
            "subjects_dir": tmp_path,
            "spacing": "oct5",
            "add_dist": False,
            "n_jobs": 2,
        },
        "bem_model": {
            "subject": "sub-01",
            "ico": 4,
            "conductivity": (0.3,),
            "subjects_dir": tmp_path,
        },
        "bem_solution": "bem-model",
        "forward": {
            "info": data.info,
            "trans": "head-to-mri",
            "src": "surface-source",
            "bem": "bem-solution",
            "meg": True,
            "eeg": False,
            "mindist": 3.0,
            "n_jobs": 2,
            "verbose": True,
            "ignore_ref": False,
        },
    }


@pytest.mark.unit
def test_forward_solution_configures_volumetric_model_and_eeg_only(
    monkeypatch, tmp_path
):
    calls = {}
    data = SimpleNamespace(info=object())
    forward = {"src": "filtered-volume-source"}

    def fake_setup_volume_source_space(**kwargs):
        calls["source"] = kwargs
        return "volume-source"

    monkeypatch.setattr(
        mne, "setup_volume_source_space", fake_setup_volume_source_space
    )
    monkeypatch.setattr(mne, "make_bem_model", lambda **kwargs: "bem-model")
    monkeypatch.setattr(mne, "make_bem_solution", lambda model: "bem-solution")

    def fake_make_forward_solution(info, **kwargs):
        calls["forward"] = {"info": info, **kwargs}
        return forward

    monkeypatch.setattr(mne, "make_forward_solution", fake_make_forward_solution)

    result, result_src = forward_solution(
        subject="sub-02",
        subjects_dir=tmp_path,
        data=data,
        transformation_matrix="head-to-mri",
        conductivity=(0.3, 0.006, 0.3),
        source_space="volumetric",
        which_sensor_dict={"eeg": True},
    )

    assert result is forward
    assert result_src == "filtered-volume-source"
    assert calls["source"] == {
        "subject": "sub-02",
        "subjects_dir": tmp_path,
        "surface": tmp_path / "sub-02" / "bem" / "inner_skull.surf",
        "add_interpolator": True,
        "n_jobs": 1,
    }
    assert calls["forward"]["src"] == "volume-source"
    assert calls["forward"]["meg"] is False
    assert calls["forward"]["eeg"] is True


@pytest.mark.unit
def test_forward_solution_rejects_unsupported_source_space(monkeypatch, tmp_path):
    def unexpected_bem_call(**kwargs):
        pytest.fail("BEM construction must not start for an invalid source space")

    monkeypatch.setattr(mne, "make_bem_model", unexpected_bem_call)

    with pytest.raises(ValueError, match="surface.*volumetric"):
        forward_solution(
            subject="sub-03",
            subjects_dir=tmp_path,
            data=SimpleNamespace(info=object()),
            transformation_matrix="head-to-mri",
            conductivity=(0.3,),
            source_space="invalid",
            which_sensor_dict={"meg": True},
        )


@pytest.mark.integration
def test_run_recon_freesurfer_preserves_paths_and_configures_environment(
    monkeypatch, tmp_path
):
    freesurfer_home = make_fake_freesurfer(tmp_path)
    subjects_dir = tmp_path / "subjects directory"
    license_path = freesurfer_home / "license file.txt"
    license_path.write_text("test license")
    mri_path = tmp_path / "T1 image.nii.gz"
    mri_path.write_text("test MRI")
    record_path = tmp_path / "recon record.txt"
    original_path = os.environ["PATH"]
    expected_path_entries = [
        entry
        for entry in original_path.split(os.pathsep)
        if entry and entry != str(freesurfer_home / "bin")
    ]
    monkeypatch.setenv("RECON_RECORD", str(record_path))
    monkeypatch.setenv("PATH", original_path)
    monkeypatch.setenv("FREESURFER_LICENSE", "/stale/license.txt")

    result = run_recon_freesurfer(
        freesurfer_home=str(freesurfer_home),
        subjects_dir=str(subjects_dir),
        license_path=str(license_path),
        subject_id="sub 01;echo injected",
        mri_path=str(mri_path),
    )

    lines = record_path.read_text().splitlines()
    assert result is None
    assert lines[:5] == [
        "arg=-i",
        f"arg={mri_path}",
        "arg=-s",
        "arg=sub 01;echo injected",
        "arg=-all",
    ]
    recorded_env = dict(line.split("=", 1) for line in lines[5:])
    assert recorded_env == {
        "FREESURFER_HOME": str(freesurfer_home),
        "SUBJECTS_DIR": str(subjects_dir),
        "FS_LICENSE": str(license_path),
        "FREESURFER_LICENSE": str(license_path),
        "SETUP_MARKER": "loaded",
        "PATH": os.pathsep.join([str(freesurfer_home / "bin"), *expected_path_entries]),
    }


@pytest.mark.integration
def test_run_recon_freesurfer_propagates_recon_all_failure(monkeypatch, tmp_path):
    freesurfer_home = make_fake_freesurfer(tmp_path, exit_code=7)
    record_path = tmp_path / "failed recon.txt"
    monkeypatch.setenv("RECON_RECORD", str(record_path))

    with pytest.raises(subprocess.CalledProcessError) as error:
        run_recon_freesurfer(
            freesurfer_home=str(freesurfer_home),
            subjects_dir=str(tmp_path / "subjects"),
            license_path=str(freesurfer_home / "license.txt"),
            subject_id="sub-01",
            mri_path=str(tmp_path / "T1.nii.gz"),
        )

    assert error.value.returncode == 7
    assert record_path.is_file()


@pytest.mark.integration
def test_run_recon_freesurfer_stops_when_setup_fails(monkeypatch, tmp_path):
    freesurfer_home = make_fake_freesurfer(tmp_path)
    (freesurfer_home / "SetUpFreeSurfer.sh").write_text("return 9\n")
    record_path = tmp_path / "unexpected recon.txt"
    monkeypatch.setenv("RECON_RECORD", str(record_path))

    with pytest.raises(subprocess.CalledProcessError) as error:
        run_recon_freesurfer(
            freesurfer_home=str(freesurfer_home),
            subjects_dir=str(tmp_path / "subjects"),
            license_path=str(freesurfer_home / "license.txt"),
            subject_id="sub-01",
            mri_path=str(tmp_path / "T1.nii.gz"),
        )

    assert error.value.returncode == 9
    assert not record_path.exists()


def make_tsss_record():
    return {
        "creator": "MaxFilter",
        "max_info": {
            "sss_info": {"in_order": 8, "out_order": 3},
            "max_st": {"buflen": 10.0, "subspcorr": 0.98},
        },
    }


@pytest.mark.unit
def test_make_bem_model_does_not_repeat_explicit_preflood(monkeypatch, tmp_path):
    attempted_prefloods = []

    def run_watershed(*, preflood, **kwargs):
        attempted_prefloods.append(preflood)
        mne_logger = logging.getLogger("mne")
        if preflood == 10:
            mne_logger.debug("before Erosion-Dilation 20.0%")
            mne_logger.debug("Fine Segmentation....20 iterations")
        else:
            mne_logger.debug("before Erosion-Dilation 0.1%")
            mne_logger.debug("Fine Segmentation....20 iterations")

    monkeypatch.setattr(sl.mne.bem, "make_watershed_bem", run_watershed)

    result = make_bem_model(
        subject="sub-01",
        subjects_dir=tmp_path / "subjects",
        bem_log_path=tmp_path / "logs" / "bem.log",
        preflood=10,
        preflood_parameter_space=(10, 15),
    )

    assert attempted_prefloods == [10, 15]
    assert result == {"preflood": 15, "erosion_pct": 0.1, "iterations": 20}


@pytest.mark.unit
def test_make_bem_model_returns_metrics_from_first_valid_attempt(monkeypatch, tmp_path):
    calls = []

    def run_watershed(**kwargs):
        calls.append(kwargs)
        mne_logger = logging.getLogger("mne")
        mne_logger.debug("before Erosion-Dilation 0.1%")
        mne_logger.debug("Fine Segmentation....42 iterations")

    monkeypatch.setattr(sl.mne.bem, "make_watershed_bem", run_watershed)

    result = make_bem_model(
        subject="sub-01",
        subjects_dir=tmp_path / "subjects",
        bem_log_path=tmp_path / "logs" / "bem.log",
        preflood=20,
        gcaatlas=False,
        volume="T2",
    )

    assert result == {"preflood": 20, "erosion_pct": 0.1, "iterations": 42}
    assert calls == [
        {
            "subject": "sub-01",
            "subjects_dir": tmp_path / "subjects",
            "overwrite": True,
            "gcaatlas": False,
            "volume": "T2",
            "preflood": 20,
            "verbose": "debug",
        }
    ]


@pytest.mark.unit
def test_make_bem_model_retries_suspect_erosion_with_excessive_iterations(
    monkeypatch, tmp_path
):
    attempted_prefloods = []

    def run_watershed(*, preflood, **kwargs):
        attempted_prefloods.append(preflood)
        mne_logger = logging.getLogger("mne")
        if preflood is None:
            mne_logger.debug("before Erosion-Dilation 0.5%")
            mne_logger.debug("Fine Segmentation....101 iterations")
        else:
            mne_logger.debug("before Erosion-Dilation 0.1%")
            mne_logger.debug("Fine Segmentation....101 iterations")

    monkeypatch.setattr(sl.mne.bem, "make_watershed_bem", run_watershed)

    result = make_bem_model(
        subject="sub-01",
        subjects_dir=tmp_path / "subjects",
        bem_log_path=tmp_path / "bem.log",
        preflood_parameter_space=(10, 15),
    )

    assert attempted_prefloods == [None, 10]
    assert result == {"preflood": 10, "erosion_pct": 0.1, "iterations": 101}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("erosion_pct", "iterations"),
    [(15.0, 100), (0.2, 101), (0.3, 100)],
)
def test_make_bem_model_accepts_values_at_quality_thresholds(
    monkeypatch, tmp_path, erosion_pct, iterations
):
    def run_watershed(**kwargs):
        mne_logger = logging.getLogger("mne")
        mne_logger.debug(f"before Erosion-Dilation {erosion_pct}%")
        mne_logger.debug(f"Fine Segmentation....{iterations} iterations")

    monkeypatch.setattr(sl.mne.bem, "make_watershed_bem", run_watershed)

    result = make_bem_model(
        subject="sub-01",
        subjects_dir=tmp_path / "subjects",
        bem_log_path=tmp_path / "bem.log",
        preflood=20,
    )

    assert result == {
        "preflood": 20,
        "erosion_pct": erosion_pct,
        "iterations": iterations,
    }


@pytest.mark.unit
def test_make_bem_model_raises_after_every_preflood_fails(monkeypatch, tmp_path):
    attempted_prefloods = []

    def run_watershed(*, preflood, **kwargs):
        attempted_prefloods.append(preflood)
        logging.getLogger("mne").debug("before Erosion-Dilation 16.0%")

    monkeypatch.setattr(sl.mne.bem, "make_watershed_bem", run_watershed)

    with pytest.raises(RuntimeError, match="all faulty"):
        make_bem_model(
            subject="sub-01",
            subjects_dir=tmp_path / "subjects",
            bem_log_path=tmp_path / "bem.log",
            preflood=None,
            preflood_parameter_space=(10, 15),
        )

    assert attempted_prefloods == [None, 10, 15]


@pytest.mark.unit
def test_make_bem_model_rejects_log_without_erosion_metric(monkeypatch, tmp_path):
    def run_watershed(**kwargs):
        logging.getLogger("mne").debug("Fine Segmentation....42 iterations")

    monkeypatch.setattr(sl.mne.bem, "make_watershed_bem", run_watershed)
    log_path = tmp_path / "bem.log"

    with pytest.raises(RuntimeError, match="Could not find erosion percentage"):
        make_bem_model(
            subject="sub-01",
            subjects_dir=tmp_path / "subjects",
            bem_log_path=log_path,
        )

    assert "Fine Segmentation" in log_path.read_text()


@pytest.mark.integration
def test_save_cov_figures_writes_both_outputs_and_closes_figures(tmp_path, caplog):
    figures = [plt.figure(), plt.figure()]
    plot_calls = []

    def plot_covariance(info, *, show):
        plot_calls.append((info, show))
        return figures

    info = {"description": "test info"}
    output_dir = tmp_path / "covariance"
    test_logger = logging.getLogger("test.save-cov")

    with caplog.at_level(logging.INFO, logger="test.save-cov"):
        save_cov_figures(
            SimpleNamespace(plot=plot_covariance),
            info,
            output_dir,
            subject="sub-01",
            tag="data",
            logger=test_logger,
        )

    assert plot_calls == [(info, False)]
    for kind, figure in zip(("matrix", "svd"), figures):
        output = output_dir / f"sub-01_data_{kind}.png"
        assert output.is_file()
        assert output.stat().st_size > 0
        assert figure.number not in plt.get_fignums()
    assert "Saved data covariance figures" in caplog.text


@pytest.mark.integration
def test_save_bem_figure_forwards_plot_options_and_closes_figure(monkeypatch, tmp_path):
    figure = plt.figure()
    plot_calls = []

    def plot_bem(**kwargs):
        plot_calls.append(kwargs)
        return figure

    monkeypatch.setattr(sl.mne.viz, "plot_bem", plot_bem)
    output_dir = tmp_path / "bem"

    save_bem_figure(
        subject="sub-01",
        subjects_dir=tmp_path / "subjects",
        out_dir=output_dir,
        orientation="sagittal",
        slices=[10, 20],
    )

    assert plot_calls == [
        {
            "subject": "sub-01",
            "subjects_dir": tmp_path / "subjects",
            "brain_surfaces": "white",
            "orientation": "sagittal",
            "slices": [10, 20],
            "show": False,
        }
    ]
    output = output_dir / "sub-01_bem_sagittal.png"
    assert output.is_file()
    assert output.stat().st_size > 0
    assert figure.number not in plt.get_fignums()


@pytest.mark.unit
def test_set_freesurfer_paths_is_idempotent(monkeypatch, tmp_path):
    freesurfer_home = tmp_path / "freesurfer"
    subjects_dir = tmp_path / "subjects"
    license_path = freesurfer_home / "license.txt"
    original_path = os.pathsep.join(["/usr/local/bin", "/usr/bin"])
    monkeypatch.setenv("PATH", original_path)
    monkeypatch.delenv("FREESURFER_LICENSE", raising=False)

    for _ in range(2):
        set_freesurfer_paths(str(freesurfer_home), str(subjects_dir), str(license_path))

    expected_bin = str(freesurfer_home / "bin")
    assert os.environ["FREESURFER_HOME"] == str(freesurfer_home)
    assert os.environ["SUBJECTS_DIR"] == str(subjects_dir)
    assert os.environ["FS_LICENSE"] == str(license_path)
    assert os.environ["FREESURFER_LICENSE"] == str(license_path)
    assert os.environ["PATH"].split(os.pathsep) == [
        expected_bin,
        "/usr/local/bin",
        "/usr/bin",
    ]


@pytest.mark.integration
def test_check_freesurfer_discovers_executable_and_configures_environment(
    monkeypatch, tmp_path
):
    freesurfer_home = tmp_path / "freesurfer"
    freesurfer_bin = freesurfer_home / "bin"
    freesurfer_bin.mkdir(parents=True)
    recon_all = freesurfer_bin / "recon-all"
    recon_all.write_text("#!/bin/sh\nexit 0\n")
    recon_all.chmod(0o755)
    license_path = freesurfer_home / "license.txt"
    license_path.write_text("test license")
    monkeypatch.setenv(
        "PATH", os.pathsep.join([str(freesurfer_bin), "/usr/local/bin", "/usr/bin"])
    )
    monkeypatch.delenv("FREESURFER_HOME", raising=False)
    monkeypatch.delenv("FS_LICENSE", raising=False)
    monkeypatch.delenv("FREESURFER_LICENSE", raising=False)

    discovered_home = check_freesurfer()

    assert discovered_home == str(freesurfer_home.resolve())
    assert os.environ["FREESURFER_HOME"] == str(freesurfer_home.resolve())
    assert os.environ["FS_LICENSE"] == str(license_path)
    assert os.environ["FREESURFER_LICENSE"] == str(license_path)
    assert os.environ["PATH"].split(os.pathsep).count(str(freesurfer_bin)) == 1


@pytest.mark.integration
def test_check_freesurfer_resolves_symlinked_executable(monkeypatch, tmp_path):
    freesurfer_home = tmp_path / "freesurfer"
    freesurfer_bin = freesurfer_home / "bin"
    freesurfer_bin.mkdir(parents=True)
    recon_all = freesurfer_bin / "recon-all"
    recon_all.write_text("#!/bin/sh\nexit 0\n")
    recon_all.chmod(0o755)
    (freesurfer_home / "license.txt").write_text("test license")
    shim_bin = tmp_path / "shims"
    shim_bin.mkdir()
    (shim_bin / "recon-all").symlink_to(recon_all)
    monkeypatch.setenv(
        "PATH", os.pathsep.join([str(shim_bin), "/usr/local/bin", "/usr/bin"])
    )

    discovered_home = check_freesurfer()

    assert discovered_home == str(freesurfer_home.resolve())


@pytest.mark.integration
def test_check_freesurfer_rejects_installation_without_license(monkeypatch, tmp_path):
    freesurfer_bin = tmp_path / "freesurfer" / "bin"
    freesurfer_bin.mkdir(parents=True)
    recon_all = freesurfer_bin / "recon-all"
    recon_all.write_text("#!/bin/sh\nexit 0\n")
    recon_all.chmod(0o755)
    monkeypatch.setenv(
        "PATH", os.pathsep.join([str(freesurfer_bin), "/usr/local/bin", "/usr/bin"])
    )

    with pytest.raises(RuntimeError, match="license not found"):
        check_freesurfer()


@pytest.mark.unit
def test_produce_aparc_processes_only_unfinished_subject_directories(
    monkeypatch, tmp_path
):
    subjects_dir = tmp_path / "subjects"
    completed_output = subjects_dir / "sub-complete" / "mri" / "aparc.a2009s+aseg.mgz"
    completed_output.parent.mkdir(parents=True)
    completed_output.write_text("already complete")
    (subjects_dir / "sub-pending").mkdir()
    (subjects_dir / "README.txt").write_text("not a subject")
    freesurfer_home = tmp_path / "freesurfer"
    license_path = freesurfer_home / "license.txt"
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join([str(freesurfer_home / "bin"), "/usr/local/bin", "/usr/bin"]),
    )
    monkeypatch.setenv("FREESURFER_LICENSE", "/stale/license.txt")
    calls = []

    def record_run(command, *, env, check):
        calls.append((command, env, check))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sl.subprocess, "run", record_run)

    produce_aparc_a2009s_aseg(
        str(subjects_dir), str(freesurfer_home), str(license_path)
    )

    assert len(calls) == 1
    command, env, check = calls[0]
    assert command == ["mri_aparc2aseg", "--s", "sub-pending", "--a2009s"]
    assert check is True
    assert env["FREESURFER_HOME"] == str(freesurfer_home)
    assert env["SUBJECTS_DIR"] == str(subjects_dir)
    assert env["FS_LICENSE"] == str(license_path)
    assert env["FREESURFER_LICENSE"] == str(license_path)
    assert env["PATH"].split(os.pathsep) == [
        str(freesurfer_home / "bin"),
        "/usr/local/bin",
        "/usr/bin",
    ]


@pytest.mark.unit
def test_prepare_template_validates_demographics_before_segmentation(
    monkeypatch, tmp_path
):
    def unexpected_segmentation(**kwargs):
        raise AssertionError("segmentation must not run without demographics")

    monkeypatch.setattr(sl, "produce_aparc_a2009s_aseg", unexpected_segmentation)
    missing_demographics = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Demographic file not found"):
        prepare_template(
            "sub-01",
            str(missing_demographics),
            SL_source_space="volumetric",
            parcellation_parc="aparc.a2009s",
            freesurfer_template_path=str(tmp_path / "templates"),
            freesurfer_home=str(tmp_path / "freesurfer"),
            freesurfer_license=str(tmp_path / "license.txt"),
        )


@pytest.mark.unit
def test_prepare_template_validates_subject_row_before_segmentation(
    monkeypatch, tmp_path
):
    demographics = tmp_path / "participants.csv"
    demographics.write_text("participant_id,age\nsub-02,1.0\n")
    calls = []
    monkeypatch.setattr(
        sl, "produce_aparc_a2009s_aseg", lambda **kwargs: calls.append(kwargs)
    )

    with pytest.raises(KeyError):
        prepare_template(
            "sub-01",
            demographics,
            SL_source_space="volumetric",
            parcellation_parc="aparc.a2009s",
            freesurfer_template_path=str(tmp_path / "templates"),
            freesurfer_home=str(tmp_path / "freesurfer"),
            freesurfer_license=str(tmp_path / "license.txt"),
        )

    assert calls == []


@pytest.mark.integration
def test_prepare_template_converts_age_in_years_and_selects_nearest_template(
    tmp_path,
):
    demographics = tmp_path / "participants.csv"
    demographics.write_text("participant_id,age\nsub-01,1.5\n")
    templates = tmp_path / "templates"
    (templates / "ANTS12-0Months3T").mkdir(parents=True)
    (templates / "ANTS18-0Months3T").mkdir()

    template_name, subjects_dir = prepare_template(
        "sub-01",
        demographics,
        SL_source_space="surface",
        freesurfer_template_path=str(templates),
    )

    assert template_name == "ANTS18-0Months3T"
    assert subjects_dir == str(templates)


@pytest.mark.unit
def test_prepare_template_requests_destrieux_segmentation_for_volumetric_source(
    monkeypatch, tmp_path
):
    demographics = tmp_path / "participants.csv"
    demographics.write_text("participant_id,age\nsub-01,1.0\n")
    templates = tmp_path / "templates"
    (templates / "ANTS12-0Months3T").mkdir(parents=True)
    freesurfer_home = tmp_path / "freesurfer"
    license_path = freesurfer_home / "license.txt"
    calls = []

    def record_segmentation(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(sl, "produce_aparc_a2009s_aseg", record_segmentation)

    prepare_template(
        "sub-01",
        demographics,
        SL_source_space="volumetric",
        parcellation_parc="aparc.a2009s",
        freesurfer_template_path=str(templates),
        freesurfer_home=str(freesurfer_home),
        freesurfer_license=str(license_path),
    )

    assert calls == [
        {
            "save_path": str(templates),
            "freesurfer_home": str(freesurfer_home),
            "freesurfer_license": str(license_path),
        }
    ]


@pytest.mark.unit
def test_check_tsss_finds_temporal_sss_after_unrelated_history():
    meg_data = SimpleNamespace(
        info={"proc_history": [{"creator": "unrelated"}, make_tsss_record()]}
    )

    assert check_tsss(meg_data)


@pytest.mark.unit
def test_check_tsss_rejects_spatial_sss_without_temporal_parameters():
    meg_data = SimpleNamespace(
        info={
            "proc_history": [
                {
                    "creator": "MaxFilter",
                    "max_info": {
                        "sss_info": {"in_order": 8, "out_order": 3},
                    },
                }
            ]
        }
    )

    assert not check_tsss(meg_data)


@pytest.mark.unit
@pytest.mark.parametrize(
    "max_st",
    [
        {"buflen": 0.0, "subspcorr": 0.98},
        {"buflen": 10.0},
        {"buflen": 10.0, "subspcorr": None},
    ],
)
def test_check_tsss_rejects_incomplete_temporal_parameters(max_st):
    record = make_tsss_record()
    record["max_info"]["max_st"] = max_st

    assert not check_tsss(SimpleNamespace(info={"proc_history": [record]}))


@pytest.mark.unit
def test_check_tsss_returns_false_without_processing_history():
    assert not check_tsss(SimpleNamespace(info={}))


@pytest.mark.unit
def test_capture_mne_log_supports_filename_without_parent_directory(
    tmp_path, monkeypatch
):
    mne_logger = logging.getLogger("mne")
    previous_level = mne_logger.level
    monkeypatch.chdir(tmp_path)

    with capture_mne_log("mne.log", level=logging.INFO) as log_path:
        mne_logger.info("captured source-localization message")

    assert "captured source-localization message" in (tmp_path / log_path).read_text()
    assert mne_logger.level == previous_level


@pytest.mark.unit
def test_capture_mne_log_restores_logger_after_exception(tmp_path):
    mne_logger = logging.getLogger("mne")
    previous_level = mne_logger.level
    previous_handlers = tuple(mne_logger.handlers)

    with pytest.raises(RuntimeError, match="stop processing"):
        with capture_mne_log(tmp_path / "logs" / "mne.log", level=logging.INFO):
            raise RuntimeError("stop processing")

    assert mne_logger.level == previous_level
    assert tuple(mne_logger.handlers) == previous_handlers


@pytest.mark.integration
def test_numpy_to_mne_epoch_preserves_source_time_courses_and_metadata():
    source_time_courses = np.arange(24.0).reshape(2, 3, 4)

    epochs = numpy_to_mne_epoch(
        source_time_courses,
        labels=["frontal-lh", "frontal-rh", "occipital-lh"],
        ch_name="misc",
        sampling_rate=200.0,
    )

    np.testing.assert_array_equal(epochs.get_data(), source_time_courses)
    assert epochs.ch_names == ["frontal-lh", "frontal-rh", "occipital-lh"]
    assert epochs.get_channel_types() == ["misc", "misc", "misc"]
    assert epochs.info["sfreq"] == pytest.approx(200.0)


@pytest.mark.unit
def test_regularized_cov_condition_applies_auto_scaled_shrinkage():
    data = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])

    covariance, condition_before, condition_after = regularized_cov_condition(
        data, shrinkage=0.25
    )

    np.testing.assert_allclose(
        covariance,
        [[4.0 / 3.0, -0.5], [-0.5, 4.0 / 3.0]],
    )
    assert condition_before == pytest.approx(3.0)
    assert condition_after == pytest.approx(2.2)


@pytest.mark.unit
def test_regularized_cov_condition_stabilizes_singular_covariance():
    data = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    _, condition_before, condition_after = regularized_cov_condition(
        data, shrinkage=0.1
    )

    assert np.isfinite(condition_after)
    assert condition_after < condition_before


@pytest.mark.unit
def test_check_digitization_points_counts_each_point_kind():
    raw = SimpleNamespace(
        info={
            "dig": [
                {"kind": FIFF.FIFFV_POINT_EXTRA},
                {"kind": FIFF.FIFFV_POINT_EXTRA},
                {"kind": FIFF.FIFFV_POINT_CARDINAL},
                {"kind": FIFF.FIFFV_POINT_HPI},
                {"kind": FIFF.FIFFV_POINT_EEG},
                {"kind": FIFF.FIFFV_POINT_EEG},
            ]
        }
    )

    counts = check_digitization_points(raw, logging.getLogger("test.dig"))

    assert counts == (2, 1, 1, 2)


@pytest.mark.unit
def test_check_digitization_points_warns_when_information_is_absent(caplog):
    raw = SimpleNamespace(info={"dig": None})

    with caplog.at_level(logging.WARNING, logger="test.missing-dig"):
        counts = check_digitization_points(raw, logging.getLogger("test.missing-dig"))

    assert counts == (0, 0, 0, 0)
    assert "No dig info at all" in caplog.text


@pytest.mark.unit
def test_build_template_index_parses_months_and_years(tmp_path):
    for name in ["ANTS6-0Months3T", "ANTS1-5Years3T", "ANTS2-0Year3T"]:
        (tmp_path / name).mkdir()
    (tmp_path / "ANTS-invalid").mkdir()
    (tmp_path / "fsaverage").mkdir()
    (tmp_path / "ANTS9-0Months3T").write_text("not a template directory")

    index = build_template_index(tmp_path)

    assert index == {
        "ANTS6-0Months3T": 6.0,
        "ANTS1-5Years3T": 18.0,
        "ANTS2-0Year3T": 24.0,
    }


@pytest.mark.unit
def test_nearest_template_dir_selects_closest_available_age(tmp_path):
    for name in ["ANTS6-0Months3T", "ANTS1-5Years3T", "ANTS2-0Years3T"]:
        (tmp_path / name).mkdir()

    name, subjects_dir = nearest_template_dir(17.0, tmp_path)

    assert name == "ANTS1-5Years3T"
    assert subjects_dir == str(tmp_path)


@pytest.mark.unit
def test_nearest_template_dir_rejects_directory_without_templates(tmp_path):
    with pytest.raises(FileNotFoundError, match="No ANTS templates"):
        nearest_template_dir(12.0, tmp_path)
