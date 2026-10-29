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
    assert "n_events" in html and "Quality control" in html
    assert "Processing and montage" not in html
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


def bootstrap_fixture():
    result = result_fixture()
    ev = result.evokeds["target"]
    result.bootstrap = SimpleNamespace(
        idps=pd.DataFrame(
            [
                dict(
                    idp=key,
                    median=10.0,
                    p05=8.0,
                    p95=12.0,
                    n_valid=9,
                    n_repeats=10,
                    status="partial",
                )
                for key in result.features.columns
            ]
        ),
        waveforms={
            "target": dict(
                ch_names=("other", "Cz"),
                times=ev.times + 1e-10,
                median=np.vstack([ev.data, ev.data]),
                p05=np.vstack([ev.data - 99e-6, ev.data - 1e-6]),
                p95=np.vstack([ev.data + 99e-6, ev.data + 2e-6]),
                n_valid=np.full((2, len(ev.times)), 10),
                n_trials=12,
            )
        },
        provenance={"n_repeats": 10},
    )
    return result


def test_bootstrap_band_uses_sensor_names_units_and_original_values():
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgba
    from meganorm.plots.plot_erp import plot_erp_result

    result = bootstrap_fixture()
    fig, axes = plot_erp_result(result)
    try:
        bands = [c for c in axes[0].collections if isinstance(c, PolyCollection)]
        assert len(bands) == 1
        points = bands[0].get_paths()[0].vertices
        assert points[:, 1].min() == pytest.approx(-1)
        assert points[:, 1].max() == pytest.approx(10)
        assert bands[0].get_facecolor()[0, :3] == pytest.approx(
            to_rgba(axes[0].lines[0].get_color())[:3]
        )
        assert axes[0].lines[0].get_ydata().max() == pytest.approx(8)
        peak = next(line for line in axes[0].lines if line.get_marker() == "o")
        assert peak.get_ydata()[0] == 42
    finally:
        plt.close(fig)


@pytest.mark.parametrize("mode", ["missing", "partial", "single", "wrong_times"])
def test_bootstrap_unavailable_or_partial_bands_flagged(mode):
    from matplotlib.collections import PolyCollection
    from meganorm.plots.plot_erp import plot_erp_result

    result = bootstrap_fixture()
    wave = result.bootstrap.waveforms["target"]
    if mode == "missing":
        result.bootstrap.waveforms["target"] = None
    elif mode == "partial":
        wave["n_valid"][1, 1] = 5
    elif mode == "single":
        wave["p05"][1, 1:] = np.nan
    else:
        wave["times"] = wave["times"] + 0.001
    fig, axes = plot_erp_result(result)
    try:
        assert any("bootstrap" in t.get_text().lower() for t in axes[0].texts)
        bands = [c for c in axes[0].collections if isinstance(c, PolyCollection)]
        assert len(bands) == (1 if mode == "partial" else 0)
    finally:
        plt.close(fig)


def test_report_bootstrap_summary_and_scope(tmp_path):
    from meganorm.plots.erp_report import write_erp_report

    html = write_erp_report(
        bootstrap_fixture(), tmp_path / "bootstrap.html"
    ).read_text()
    for text in (
        "bootstrap_median",
        "bootstrap_p05",
        "bootstrap_p95",
        "bootstrap_n_valid",
        "bootstrap_n_repeats",
        "bootstrap_status",
        "partial",
        "Quality control",
        "central 90%",
        "simultaneous",
    ):
        assert text in html
    assert "Processing and montage" not in html
