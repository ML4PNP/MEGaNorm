# MEGaNorm

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Documentation Status](https://readthedocs.org/projects/meganorm/badge/?version=latest)](https://meganorm.readthedocs.io/en/latest/?badge=latest)
![Last Commit](https://img.shields.io/github/last-commit/ML4PNP/MEGaNorm.svg)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ML4PNP/MEGaNorm/main?filepath=notebooks%2F)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15441320.svg)](https://doi.org/10.5281/zenodo.15441320)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


<p align="center">
  <img src="docs/images/logo.png" alt="MEGaNorm Logo" width="180"/>
</p>

<h1 align="center">MEGaNorm</h1>

<p align="center">
  Normative modeling of EEG/MEG brain dynamics across populations and timescales.
</p>


**MEGaNorm** is a Python package that wraps [MNE-Python](https://github.com/mne-tools/mne-python) and [PCNToolkit](https://github.com/amarquand/PCNtoolkit) functionalities for extracting functional imaging-derived phenotypes (f-IDPs) from large-scale EEG and MEG datasets, and then deriving their normative ranges. It allows researchers to analyze large MEG and EEG dataset using high-performance computing facilities, and then build, visualize, and analyze normative models of brain dynamics across individuals.

![Overview](docs/images/pipeline_overview.png)

&#x20;

---

## 🚀 Features

* Compatibility with MNE-Python, PCNToolkit, SpecParam libraries
* Normative modeling of oscillatory brain activity
* Using high performance capabilities on SLURM clusters 
* EEG and MEG support with BIDS integration
* Ready for reproducible deployment with Docker

---

## 📦 Installation

### Option 1: From PyPI (recommended)

This is the easiest way to get started if you just want to use the toolbox.

```bash
conda create --channel=conda-forge --strict-channel-priority --name mne python=3.12 mne

conda activate mne

pip install meganorm
```

---

### Option 2: Installl from the source files

```bash
# 1. Create and activate environment
conda create --channel=conda-forge --strict-channel-priority --name mne python=3.12 mne

conda activate mne

# 2. Clone and install MEGaNorm
git clone https://github.com/ML4PNP/MEGaNorm.git
cd MEGaNorm/
pip install .
```

---

### Option 3: Using Docker

We provide a pre-configured Docker environment with Jupyter Lab. You can either **build the image locally** or **pull the latest version from Docker Hub**.

#### Option A: Build the image locally

```bash
make build
make run
```

#### Option B: Pull the latest image from Docker Hub

```bash
make pull
make run
```

This mounts:

* `notebooks/` → for saving Jupyter notebooks
* `results/` → for analysis outputs
* `data/` → for raw/processed EEG/MEG data

Jupyter will open in your browser on [http://localhost:8888](http://localhost:8888)

---

## 📒 Getting Started (under construction)

👉 **documentation** Early helpers are available at: [https://meganorm.readthedocs.io](https://meganorm.readthedocs.io). We are working hard to add more thorough documentations and tutorials and they will be available soon.

```python 
import meganorm
```

Explore examples in the [`notebooks/`](notebooks/) folder.

---

## 🧚‍♂️ Testing (under construction)

Run unit tests using:

```bash
pytest tests/
```

---

## 🧠 Citing MEGaNorm

If you use MEGaNorm in your work, please cite:

**DOI**: [10.5281/zenodo.15441320](https://doi.org/10.5281/zenodo.15441320)

You can download BibTeX and other citation formats directly from the [Zenodo page](https://doi.org/10.5281/zenodo.15441320).


---

## 🤝 Contributing

Contributions, issues and feature requests are welcome!

See [CONTRIBUTING.md](CONTRIBUTING.md) for more info.

---

## 📜 License

This project is licensed under the terms of the **GNU General Public License v3.0** – see the [LICENSE](LICENSE) file for details.
