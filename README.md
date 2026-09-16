# MEGaNorm

[![PyPI](https://img.shields.io/pypi/v/meganorm?logo=pypi&logoColor=white&color=3775A9)](https://pypi.org/project/meganorm/)
[![Python](https://img.shields.io/pypi/pyversions/meganorm?logo=python&logoColor=white&color=3776AB)](https://pypi.org/project/meganorm/)
[![Documentation](https://readthedocs.org/projects/meganorm/badge/?version=latest)](https://meganorm.readthedocs.io/en/latest/)
[![Docker Pulls](https://img.shields.io/docker/pulls/smkia/meganorm?logo=docker&logoColor=white&color=2496ED)](https://hub.docker.com/r/smkia/meganorm)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ML4PNP/MEGaNorm/main?filepath=notebooks%2F)
[![License](https://img.shields.io/github/license/ML4PNP/MEGaNorm?color=blue)](https://github.com/ML4PNP/MEGaNorm/blob/main/LICENSE)
[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15441320.svg)](https://doi.org/10.5281/zenodo.15441320)
[![Paper DOI](https://img.shields.io/badge/DOI-10.1038%2Fs42003--026--09825--2-B31B1B?logo=doi&logoColor=white)](https://doi.org/10.1038/s42003-026-09825-2)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/14128/badge)](https://www.bestpractices.dev/projects/14128)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/meganorm?period=month&units=international_system&left_text=downloads/month&left_color=grey&right_color=brightgreen)](https://pypi.org/project/meganorm/)
[![Last Commit](https://img.shields.io/github/last-commit/ML4PNP/MEGaNorm?logo=github&color=informational)](https://github.com/ML4PNP/MEGaNorm/commits/main)

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
conda create --channel=conda-forge --strict-channel-priority --name meganorm python=3.12

conda activate meganorm

pip install meganorm
```

---

### Option 2: Installl from the source files

```bash
# 1. Create and activate environment
conda create --channel=conda-forge --strict-channel-priority --name meganorm python=3.12

conda activate meganorm

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
make pull # or docker pull smkia/meganorm:latest
make run
```

This mounts:

* `notebooks/` → for saving Jupyter notebooks
* `results/` → for analysis outputs
* `data/` → for raw/processed EEG/MEG data

Jupyter will open in your browser on [http://localhost:8888](http://localhost:8888)


---
## FreeSurfer installation (necessary for source localization)
If you plan to use the MEGaNorm package for source localization, you must first download and install FreeSurfer from [https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall). You will also need to obtain a FreeSurfer license key and provide it to the MEGaNorm package.

After installing FreeSurfer, you need to run the FreeSurfer cortical reconstruction pipeline on your MRI data using the following commands:

```bash
export FREESURFER_HOME=/usr/local/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=/path/to/your/subjects

recon-all -s sub-01 -i /path/to/sub-01_T1w.nii.gz -all
```

For more details on the recon-all pipeline, please refer to the FreeSurfer recon-all documentation: [https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all](https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all)

---

## Getting Started (under construction)

👉 **documentation** Early helpers are available at: [https://meganorm.readthedocs.io](https://meganorm.readthedocs.io). We are working hard to add more thorough documentations and tutorials and they will be available soon.

```python 
import meganorm
```

Explore examples in the [`notebooks/`](notebooks/) folder.

---

## Testing (under construction)

Run unit tests using:

```bash
pytest tests/
```

---

## Citing MEGaNorm

* Citing the package (**DOI**: [10.5281/zenodo.15441320](https://doi.org/10.5281/zenodo.15441320)): You can download BibTeX and other citation formats directly from the [Zenodo page](https://doi.org/10.5281/zenodo.15441320): Zamanzadeh, M., Verduyn, Y., & Kia, S. M. (2025). MEGaNorm: a Python package for normative modeling on MEG and EEG data (v0.1.0). Zenodo. 

* Citing the paper ([nature communication biology](https://doi.org/10.1038/s42003-026-09825-2)): Zamanzadeh, Mohammad, Ymke Verduyn, Augustijn de Boer, Tomas Ros, Thomas Wolfers, Richard Dinga, Marie Šafář Postma, Andre F. Marquand, Marijn van Wingerden, and Seyed Mostafa Kia. "Normative modeling of MEG brain oscillations across the human lifespan." Communications biology (2026). 

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome!

See [CONTRIBUTING.md](CONTRIBUTING.md) for more info.

---

## 📜 License

This project is licensed under the terms of the **GNU General Public License v3.0** – see the [LICENSE](LICENSE) file for details.
