# MEGaNorm

[![PyPI](https://img.shields.io/pypi/v/meganorm?logo=pypi\&logoColor=white\&color=3775A9)](https://pypi.org/project/meganorm/)
[![Python](https://img.shields.io/pypi/pyversions/meganorm?logo=python\&logoColor=white\&color=3776AB)](https://pypi.org/project/meganorm/)
[![Tests](https://github.com/ML4PNP/MEGaNorm/actions/workflows/tests.yml/badge.svg)](https://github.com/ML4PNP/MEGaNorm/actions/workflows/tests.yml)
[![Documentation](https://readthedocs.org/projects/meganorm/badge/?version=latest)](https://meganorm.readthedocs.io/en/latest/)
[![Docker Pulls](https://img.shields.io/docker/pulls/smkia/meganorm?logo=docker\&logoColor=white\&color=2496ED)](https://hub.docker.com/r/smkia/meganorm)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ML4PNP/MEGaNorm/main?filepath=notebooks%2F)
[![License](https://img.shields.io/github/license/ML4PNP/MEGaNorm?color=blue)](https://github.com/ML4PNP/MEGaNorm/blob/main/LICENSE)
[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21291858.svg)](https://doi.org/10.5281/zenodo.21291858)
[![Paper DOI](https://img.shields.io/badge/DOI-10.1038%2Fs42003--026--09825--2-B31B1B?logo=doi\&logoColor=white)](https://doi.org/10.1038/s42003-026-09825-2)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/14128/badge)](https://www.bestpractices.dev/projects/14128)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/meganorm?period=month\&units=international_system\&left_text=downloads/month\&left_color=grey\&right_color=brightgreen)](https://pypi.org/project/meganorm/)
[![Last Commit](https://img.shields.io/github/last-commit/ML4PNP/MEGaNorm?logo=github\&color=informational)](https://github.com/ML4PNP/MEGaNorm/commits/main)

<p align="center">
  <img src="docs/images/logo.png" alt="MEGaNorm logo" width="180"/>
</p>

<h1 align="center">MEGaNorm</h1>

<p align="center">
  Normative modeling of EEG/MEG brain dynamics across populations and timescales.
</p>

**MEGaNorm** is a Python package for extracting functional imaging-derived phenotypes (f-IDPs) from large-scale EEG and MEG datasets and deriving their normative ranges. It integrates functionality from [MNE-Python](https://github.com/mne-tools/mne-python), [PCNToolkit](https://github.com/amarquand/PCNtoolkit), and [SpecParam](https://github.com/fooof-tools/fooof) into workflows for large-scale analysis of electrophysiological data.

MEGaNorm supports processing on high-performance computing (HPC) infrastructure and provides tools for building, visualizing, and analyzing normative models of brain dynamics across individuals and populations.

![Overview of the MEGaNorm pipeline](docs/images/pipeline_overview.png)

---

## Features

* Extraction of electrophysiological functional imaging-derived phenotypes (f-IDPs)
* Normative modeling of oscillatory brain activity
* Integration with MNE-Python, PCNToolkit, and SpecParam
* EEG and MEG support with BIDS integration
* High-performance computing workflows for SLURM clusters
* Reproducible deployment using Docker

---

## Installation

### From PyPI (recommended)

For most users, installation from PyPI is the recommended option.

```bash
conda create --channel=conda-forge --strict-channel-priority --name meganorm python=3.12
conda activate meganorm
pip install meganorm
```

### From source

To install the latest development version:

```bash
# Create and activate the environment
conda create --channel=conda-forge --strict-channel-priority --name meganorm python=3.12
conda activate meganorm

# Clone and install MEGaNorm
git clone https://github.com/ML4PNP/MEGaNorm.git
cd MEGaNorm
pip install .
```

### Using Docker

A pre-configured Docker environment with JupyterLab is available. You can either build the image locally or pull the latest image from Docker Hub.

#### Build locally

```bash
make build
make run
```

#### Pull from Docker Hub

```bash
make pull
make run
```

Alternatively:

```bash
docker pull smkia/meganorm:latest
```

The Docker environment mounts:

* `notebooks/` for Jupyter notebooks
* `results/` for analysis outputs
* `data/` for EEG/MEG data

JupyterLab is available at http://localhost:8888.

---

## FreeSurfer

FreeSurfer is required only when using MEGaNorm workflows that involve source localization.

Download and installation instructions are available in the [FreeSurfer documentation](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall). A FreeSurfer license is also required.

After installation, cortical reconstruction of the anatomical MRI can be performed using `recon-all`, for example:

```bash
export FREESURFER_HOME=/usr/local/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=/path/to/your/subjects

recon-all -s sub-01 -i /path/to/sub-01_T1w.nii.gz -all
```

See the [FreeSurfer `recon-all` documentation](https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all) for further information.

---

## Getting started

The full MEGaNorm documentation, including usage instructions and examples, is available at [meganorm.readthedocs.io](https://meganorm.readthedocs.io/).

After installation, verify that MEGaNorm can be imported:

```python
import meganorm
```

Example workflows are also available in the [`notebooks/`](notebooks/) directory.

---

## Testing

MEGaNorm includes an automated test suite covering its core processing,
feature-extraction, normative-modeling, source-localization, IO, layout,
plotting, and utility functionality. The full suite runs automatically through
[GitHub Actions](https://github.com/ML4PNP/MEGaNorm/actions/workflows/tests.yml)
on pushes and pull requests.

To run the tests locally from a development checkout:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Please report unexpected behavior or reproducibility issues through the
[GitHub issue tracker](https://github.com/ML4PNP/MEGaNorm/issues).

---

## Citation

If you use MEGaNorm in your research, please cite the software. Depending on your use of the package, you may also cite the associated scientific publication describing the MEGaNorm framework and its application to lifespan MEG normative modeling.

### Software

The citation metadata for MEGaNorm is provided in [`CITATION.cff`](CITATION.cff) and is also available through the **Cite this repository** option on GitHub.

The software is archived on Zenodo:

**Zamanzadeh, M., Verduyn, Y., & Kia, S. M.** *MEGaNorm: A Python package for normative modeling of MEG and EEG data.* Zenodo.
https://doi.org/10.5281/zenodo.21291858

### Scientific publication

For the scientific framework and its application to normative modeling of brain oscillations across the human lifespan, please cite:

**Zamanzadeh, M., Verduyn, Y., de Boer, A., Ros, T., Wolfers, T., Dinga, R., Šafář Postma, M., Marquand, A. F., van Wingerden, M., & Kia, S. M.** (2026). Normative modeling of MEG brain oscillations across the human lifespan. *Communications Biology*.
https://doi.org/10.1038/s42003-026-09825-2

---

## Contributing

Contributions, bug reports, and feature requests are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution guidelines.

---

## License

MEGaNorm is distributed under the **GNU General Public License v3.0**. See [`LICENSE`](LICENSE) for details.
