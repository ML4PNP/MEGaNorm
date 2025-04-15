# MEGaNorm
## Installation

To install **MEGaNorm**, we recommend using a Conda environment for dependency isolation and reproducibility.

Follow the steps below:

```bash
# 1. Create a new environment with Python 3.12 and MNE
conda create --channel=conda-forge --strict-channel-priority --name mne python=3.12 mne

# 2. Activate the environment
conda activate mne

# 3. Navigate to the MEGaNorm project directory
cd MEGaNorm/

# 4. Install MEGaNorm using pip
pip install .
