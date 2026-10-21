# Early sensor-space ERP API

For a runnable example with synthetic data, plots and QC, see the
[ERP IDP extraction tutorial](notebooks/erp_idp_extraction_tutorial.ipynb).

The opt-in API lives in `meganorm/src/erp.py` on `dev_erp`. It provides automatic
EEG preprocessing and epoching, plus extraction from prepared EEG or MEG evoked
responses. The existing spectral pipeline and SLURM runner are unchanged.

## Multiple components from a recording

```python
import mne
from meganorm.src.erp import ERPComponent, ERPConfig, run_erp_pipeline

raw = mne.io.read_raw_fif("subject_raw.fif", preload=True)
# Ensure channel types and sensor locations describe this recording.
# Existing meganorm.src.preprocess.prepare_eeg_data(raw, path) can be used
# when its channels.tsv / montage.csv conventions match the dataset.

components = {
    "N100": ERPComponent(
        conditions=("standard", "target"), channels=("Cz", "Pz"),
        window=(0.08, 0.15), polarity="neg",
    ),
    "late_positive": ERPComponent(
        conditions=("target",), channels=("Pz",),
        window=(0.25, 0.5), polarity="pos",
        metrics=("peak_amplitude", "peak_latency", "mean_amplitude"),
    ),
}
config = ERPConfig(tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), min_trials=10)
result = run_erp_pipeline(
    raw, "sub-01", components,
    annotation_map={"standard": "Stimulus/S  5", "target": "Stimulus/S  6"},
    config=config,
)
result.features.to_csv("sub-01.csv")
result.qc.to_csv("sub-01_erp_qc.csv", index=False)
```

The windows, names and event descriptions above are illustrative protocol
choices. Add any number of components or conditions. Each component measures
individual sensors; sensor-group averages and condition contrasts are not yet
implemented. All components in a run share epoch and baseline settings.

Alternatively, pass `events=events, event_id={"standard": 5, "target": 6}`.
Events must contain integer absolute sample indices at the recording's original
sampling rate, including `raw.first_samp`. The pipeline resolves annotations
before cleaning and jointly resamples data/events when `resample_sfreq` is set.
Duplicate event samples, including duplicates introduced by resampling, fail
explicitly rather than silently merging trials.

## Output and missing values

`result.features` has one subject row and scalar columns such as
`ERP__target__N100__peak_amplitude_uV__Cz`. Peak latency uses milliseconds;
amplitude stays signed, including absolute-polarity peak searches. Window means
average samples within the inclusive window. Tied extrema select the earliest
sample. Zero-only signals do not produce a peak.

Missing conditions/sensors, marked bad sensors, nonfinite evoked windows and
insufficient trials produce NaN with a reason in `result.qc`. Peak metrics also
return NaN when the requested polarity is absent; a mean can still be computed.
QC includes retained trials and, for raw processing, event/rejection counts.
Invalid configuration or sensor-type mismatches fail explicitly.

Keep QC outside the feature-response list. Use a separate ERP result directory
when collecting subject CSVs: the existing collector replaces subject rows, so
ERP-only reruns should not overwrite an existing spectral result table.

## Prepared evoked responses

```python
from meganorm.src.erp import extract_erp_idps

features, qc = extract_erp_idps(
    "sub-01", {"standard": standard_evoked, "target": target_evoked},
    components, min_trials=10,
)
```

For MEG, specify `sensor_type="mag"` or `"grad"` and matching channel names.
Amplitudes are exported in T or T/m respectively; sensor types are never mixed.
`nave` is used as the retained-trial count, so supply ordinary condition averages
with accurate `nave`, not weighted contrasts with effective sample sizes.

## Reusable tutorial helpers

- `notebooks/_erp_demo.py:make_demo_recording(seed=42)` supplies the shared synthetic
  recording used in the notebook and tests. Keep the helper beside the notebook;
  no tests package is needed.
- `meganorm.plots.plot_erp.plot_erp(evokeds, components=..., channels=...)`
  returns a Figure and an axes array. It displays condition waveforms in the
  appropriate units and shades the configured component windows per sensor.
  It does not show or save the plot unless requested (`show=True` or `fig.savefig`).
- `meganorm.src.erp.extract_recording(path, subject_id, components, ...)`
  loads a recording using MNE and delegates to `run_erp_pipeline()`. It accepts
  either `annotation_map` or `events` plus `event_id`, and returns `ERPResult`.
  For custom channel/montage preparation, load Raw first, use the appropriate
  existing helper, and then call `run_erp_pipeline()`.

The recording wrapper belongs beside ERP orchestration in `src/erp.py`.
`utils/IO.py` remains unchanged: its existing `load_recording()` is coupled to
the broader configuration and empty-room/source-localization handling. This
placement avoids introducing an ERP pipeline dependency into that shared module.

## Preprocessing and current limits

- Uses a copy; input Raw and Evoked objects are not modified.
- Configurable EEG reference and fractional filter cutoffs. ICA fits a separate
  filtered copy, then applies the resulting exclusion to ERP-filtered data.
- Reuses MEGaNorm's `find_ica_component()` for auxiliary-channel correlation.
  With usable EOG/ECG channels, this shared helper selects the strongest
  correlated component per auxiliary channel above threshold. Without them,
  ICLabel provides automatic classification and requires sensor locations.
  Classification/filter/reference choices still require dataset validation.
- Automatic MNE peak-to-peak and flat thresholds plus existing BAD annotations
  reject epochs. The default EEG peak-to-peak threshold is 150 µV; configure it
  for the protocol. AutoReject interpolation and automatic bad-channel detection
  are not yet wired into this ERP path.
- Existing bad-channel metadata is respected. Nonfinite continuous EEG fails
  before filtering. No montage is guessed and there are no interactive steps.
- ICA exclusions and epoch drop logs are available in `preprocessing_qc`.
  Filtering/ICA defaults are starting settings, not scientifically validated
  defaults for every ERP protocol.
- Raw preprocessing currently supports EEG. Prepared-evoked extraction also
  supports MEG. No source reconstruction, bootstrap, BIDS event-table reader,
  multi-session aggregation or SLURM dispatch is added in this first increment.

## Quick synthetic checks

With MEGaNorm dependencies installed, run from the repository root:

```bash
python -m pytest -q tests/src/test_erp.py
```

The tests exercise known peaks, multiple conditions/components, units, baseline,
event alignment/resampling, rejection, missing values and CSV output. An ICA
wiring test uses a mock; a separate real Infomax smoke test checks removal of a
synthetic EOG mixture. These do not validate performance on real recordings or
the ICLabel fallback.

**Historical validation (not a current test result):** on 2026-09-15, nine tests passed in an isolated test
environment based on the local `mne-dev` environment. Its missing `pingouin`
dependency (imported by the existing normative-modeling module) was installed
only into `/tmp/meganorm-erp-test-env`. Black and `git diff --check` also passed.

The current suite contains 12 ERP test methods under `tests/src/test_erp.py`.
They are discovered by the normal pytest command. Run the command above for
current results; the historical run did not cover all current tests.
