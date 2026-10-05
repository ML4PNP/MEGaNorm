.. image:: _static/logo.png
   :alt: MEGaNorm logo
   :width: 420px
   :align: center

|

MEGaNorm
========

**Normative modeling of brain dynamics from large-scale MEG and EEG data**

MEGaNorm is an open-source Python package for extracting functional
imaging-derived phenotypes (IDPs) from large-scale MEG and EEG datasets
and constructing normative models of brain dynamics across the lifespan.

Built around established neuroimaging and normative-modeling tools,
including `MNE-Python <https://mne.tools/>`_ and
`PCNToolkit <https://pcntoolkit.readthedocs.io/>`_, MEGaNorm provides
a reproducible pipeline from electrophysiological data to individual-level
normative deviation estimates.


Quick Links
-----------

|repository_link|
| `PyPI <https://pypi.org/project/meganorm/>`_
| |documentation_link|
| |paper_link|
| |software_link|


What is MEGaNorm?
-----------------

MEGaNorm provides an integrated workflow for normative modeling of
electrophysiological brain activity. It is designed to support the
analysis of large and heterogeneous MEG and EEG datasets while
facilitating reproducible and scalable research.

The package combines preprocessing and feature extraction with
normative modeling, allowing researchers to characterize population-level
variation in brain dynamics and quantify how individual observations
deviate from expected normative patterns.

MEGaNorm builds upon the broader Python neuroimaging and normative-modeling
ecosystem, particularly `MNE-Python <https://mne.tools/>`_ for MEG/EEG
analysis and `PCNToolkit <https://pcntoolkit.readthedocs.io/>`_ for
normative modeling.


.. image:: _static/pipeline_overview.png
   :alt: Overview of the MEGaNorm workflow
   :width: 90%
   :align: center


Key Capabilities
----------------

MEGaNorm supports:

* **MEG and EEG processing** using established MNE-Python workflows.
* **Functional IDP extraction** from electrophysiological recordings.
* **Spectral characterization** of oscillatory brain activity.
* **Normative modeling** using PCNToolkit.
* **Individual deviation mapping** relative to normative reference models.
* **BIDS-compatible workflows** for standardized neuroimaging datasets.
* **Large-scale computation** including SLURM/HPC environments.
* **Containerized deployment** using Docker for reproducible analyses.


Installation
------------

MEGaNorm can be installed directly from PyPI:

.. code-block:: bash

   pip install meganorm

Alternatively, the Docker image provides a reproducible environment:

.. code-block:: bash

   docker pull smkia/meganorm:latest

For fully reproducible analyses, replace ``latest`` with a published version
from the `Docker image tags <https://hub.docker.com/r/smkia/meganorm/tags>`_.

For available releases and package metadata, see the
`MEGaNorm PyPI page <https://pypi.org/project/meganorm/>`_.


Getting Started
---------------

For a minimal example running MEGaNorm on a single recording and a local
computer, see the :doc:`Getting Started guide <getting_started>`.

For large datasets, advanced configuration, and SLURM-based processing,
see the full example notebooks:

|notebooks_link|


Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules


Citation
--------

If you use MEGaNorm in your research, please cite the software package
and the associated scientific publication where appropriate.


Citing the Package
~~~~~~~~~~~~~~~~~~

The MEGaNorm software is archived on Zenodo:

|software_doi_link|

**Recommended citation**

   Zamanzadeh, M., Verduyn, Y., & Kia, S. M.
   *MEGaNorm: A Python package for normative modeling of MEG and EEG data.* Zenodo.

BibTeX and other citation formats can be downloaded directly from the
|software_link|.

The concept DOI identifies the software across versions. For reproducibility,
select the archived version used in your analysis and cite its version DOI.


Citing the Paper
~~~~~~~~~~~~~~~~

For the scientific methodology and normative modeling framework, please
cite the associated publication in *Communications Biology*:

   Zamanzadeh, M., Verduyn, Y., de Boer, A., Ros, T., Wolfers, T.,
   Dinga, R., Šafář Postma, M., Marquand, A. F., van Wingerden, M.,
   & Kia, S. M. (2026).
   *Normative modeling of MEG brain oscillations across the human lifespan.*
   Communications Biology.

|paper_doi_link|

Acknowledgements
----------------

We gratefully acknowledge the starter grant for the **MEGaNorm** project, 
funded by the Dutch Ministry of Education, Culture and Science under the 
National Sector Plan. We further acknowledge support from the 
**NWA Innovative Projects within the Routes** grant (NWA.1418.24.006) from 
the Dutch Research Council (NWO). This work also used the Dutch national 
e-infrastructure with the support of the **SURF Cooperative** 
(EINF-8659, EINF-13793, and EINF-18102).

.. list-table::
   :widths: 33 33 33
   :align: center
   :class: funding-logos

   * - .. image:: _static/tilburg-university-logo.png
          :alt: Tilburg University
          :width: 180px
          :target: https://www.tilburguniversity.edu/

     - .. image:: _static/nwo-logo.jpg
          :alt: Dutch Research Council (NWO)
          :width: 140px
          :target: https://www.nwo.nl/projecten/vqlab92202

     - .. image:: _static/surf-logo.png
          :alt: SURF Cooperative
          :width: 140px
          :target: https://www.surf.nl/en

Developed by ML4PNP
-------------------

MEGaNorm is developed and maintained by the
`Machine Learning for Precision NeuroPsychiatry (ML4PNP) Lab
<https://ml4pnp.github.io/>`_.

.. image:: _static/ml4pnp_logo.png
   :alt: Machine Learning for Precision NeuroPsychiatry Lab
   :width: 260px
   :align: center
   :target: https://ml4pnp.github.io/


Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`