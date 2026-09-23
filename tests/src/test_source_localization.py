import logging
import os
import subprocess
from types import SimpleNamespace

import matplotlib.pyplot as plt
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
    make_bem_model,
    nearest_template_dir,
    numpy_to_mne_epoch,
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
