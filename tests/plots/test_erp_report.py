"""Reports must visualize persisted measurements, never re-extract them."""

from types import SimpleNamespace
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pytest
from meganorm.src.erp import ERPComponent, extract_erp_idps


def result_fixture():
    ev = mne.EvokedArray(
        np.array([[0, 1, 8, 2, 0]]) * 1e-6,
        mne.create_info(["Cz"], 10, "eeg"),
        tmin=0,
        nave=12,
    )
    components = {
        "P<script>": ERPComponent(
            ("target",),
            ("Cz",),
            (0.1, 0.3),
            "pos",
            ("peak_amplitude", "peak_latency", "mean_amplitude"),
        )
    }
    features, qc = extract_erp_idps("person<script>", {"target": ev}, components)
    features.iloc[0, :] = [42, 150, 23]
    qc["n_events"] = 15
    return SimpleNamespace(
        features=features,
        qc=qc,
        components=components,
        evokeds={"target": ev},
        preprocessing_qc={"montage": {"source": "template"}},
    )


def test_plot_uses_saved_values_and_does_not_mutate():
    from meganorm.plots.plot_erp import plot_erp_result

    result = result_fixture()
    before = result.features.copy(deep=True)
    data = result.evokeds["target"].data.copy()
    fig, axes = plot_erp_result(result)
    try:
        peaks = [line for line in axes[0].lines if line.get_marker() == "o"]
        assert len(peaks) == 1
        assert list(peaks[0].get_xdata()) == [150]
        assert list(peaks[0].get_ydata()) == [42]
        assert any(
            np.allclose(c.get_segments()[0][:, 1], 23) for c in axes[0].collections
        )
        pd.testing.assert_frame_equal(result.features, before)
        np.testing.assert_array_equal(result.evokeds["target"].data, data)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("mode", ["missing_metric", "bad_qc", "absent", "nonfinite"])
def test_unavailable_measurements_never_get_markers(mode):
    from meganorm.plots.plot_erp import plot_erp_result

    result = result_fixture()
    if mode == "missing_metric":
        result.features = result.features.drop(columns=result.features.columns[1])
    elif mode == "bad_qc":
        result.qc.loc[0, "status"] = "insufficient_trials"
    elif mode == "absent":
        result.evokeds["target"] = None
    else:
        result.evokeds["target"].data[:] = np.nan
    fig, axes = plot_erp_result(result)
    try:
        assert not any(line.get_marker() == "o" for line in axes[0].lines)
        if mode in ("absent", "nonfinite"):
            assert any("No usable" in t.get_text() for t in axes[0].texts)
    finally:
        plt.close(fig)


def test_report_html_escaping_measurements_export_and_closure(tmp_path):
    from meganorm.plots.erp_report import write_erp_report

    result = result_fixture()
    initial = plt.get_fignums()
    path = write_erp_report(result, tmp_path / "report.html", save_figures=True)
    html = path.read_text()
    assert "person&lt;script&gt;" in html
    assert "P&lt;script&gt;" in html
    assert "peak_amplitude" in html and "42" in html
    assert "n_events" in html and "montage" in html
    assert "person<script>" not in html
    assert len(list((tmp_path / "plots").glob("*.png"))) == 1
    assert len(list((tmp_path / "plots").glob("*.svg"))) == 1
    assert plt.get_fignums() == initial


def test_report_paginates_and_preserves_previous_html_on_failure(tmp_path, monkeypatch):
    import meganorm.plots.erp_report as reports

    result = result_fixture()
    spec = next(iter(result.components.values()))
    result.components = {
        "P": ERPComponent(
            spec.conditions, ("Cz", "Pz", "Fz", "Oz"), spec.window, spec.polarity
        )
    }
    result.features, result.qc = extract_erp_idps(
        "person", result.evokeds, result.components
    )
    report = reports.write_erp_report(
        result, tmp_path / "report.html", save_figures=True
    )
    assert len(list((tmp_path / "plots").glob("*.png"))) == 2
    previous = report.read_bytes()
    figures = plt.get_fignums()

    def fail_save(self, *args, **kwargs):
        raise OSError("disk write failed")

    monkeypatch.setattr(mne.Report, "save", fail_save)
    with pytest.raises(OSError, match="disk write failed"):
        reports.write_erp_report(result, report)
    assert report.read_bytes() == previous
    assert not list(tmp_path.glob(".report.*.html"))
    assert plt.get_fignums() == figures


def test_report_closes_figure_if_adding_image_fails(tmp_path, monkeypatch):
    from meganorm.plots.erp_report import write_erp_report

    initial = plt.get_fignums()

    def fail_add(self, *args, **kwargs):
        raise ValueError("image rejected")

    monkeypatch.setattr(mne.Report, "add_figure", fail_add)
    with pytest.raises(ValueError, match="image rejected"):
        write_erp_report(result_fixture(), tmp_path / "report.html")
    assert plt.get_fignums() == initial


def test_other_sensor_units_and_different_component_windows():
    from meganorm.plots.plot_erp import plot_erp_result

    ev = mne.EvokedArray(
        np.array([[0.0, 1.0, 8.0, 2.0, 0.0]]) * 1e-12,
        mne.create_info(["MEG"], 10, "mag"),
        tmin=0,
        nave=12,
    )
    components = {
        name: ERPComponent(("target",), ("MEG",), window, "pos", sensor_type="mag")
        for name, window in [("A", (0.1, 0.2)), ("B", (0.2, 0.3))]
    }
    features, qc = extract_erp_idps("person", {"target": ev}, components)
    result = SimpleNamespace(
        features=features, qc=qc, evokeds={"target": ev}, components=components
    )
    fig, axes = plot_erp_result(result)
    try:
        assert axes[0].get_ylabel() == "Amplitude (T)"
        assert len(axes[0].patches) == 2
        assert np.max(axes[0].lines[0].get_ydata()) == 8e-12
    finally:
        plt.close(fig)
