"""Scientific API for local electrophysiological normative modeling."""

from meganorm.utils.IO import Config
from meganorm.utils.erp import (
    ERPComponentSettings,
    ERPEventSpec,
    ERPMontageSettings,
)
from .datasets import Dataset
from .pipeline import Pipeline
from .modeling import NormativeModel
from .results import FeatureDataset, NormativeResults

__all__ = [
    "ERPResult",
    "extract_recording",
    "load_erp_result",
    "Config",
    "ERPComponentSettings",
    "ERPEventSpec",
    "ERPMontageSettings",
    "Dataset",
    "Pipeline",
    "NormativeModel",
    "FeatureDataset",
    "NormativeResults",
]

from meganorm.src.erp import ERPResult, extract_recording
from meganorm.utils.erp_outputs import load_erp_result
