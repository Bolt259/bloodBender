"""
bloodTwin - LSTM-based Blood Glucose Prediction Model

A unified LSTM model trained on multi-patient insulin pump data to predict
blood glucose dynamics over short-to-medium horizons (30-90 minutes).

Components:
- models: LSTM architecture (PyTorch Lightning)
- data: Dataset and DataLoader implementations
- pipelines: Training, evaluation, and export workflows
- configs: YAML-based configuration
- artifacts: Saved models, scalers, and exports
- analytics: Training metrics and evaluation results
"""

__version__ = "1.0.0"
__author__ = "bloodBender Team"

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
BLOODTWIN_ROOT = Path(__file__).parent

# Key directories
CONFIGS_DIR = BLOODTWIN_ROOT / "configs"
MODELS_DIR = BLOODTWIN_ROOT / "models"
DATA_DIR = BLOODTWIN_ROOT / "data"
PIPELINES_DIR = BLOODTWIN_ROOT / "pipelines"
ARTIFACTS_DIR = BLOODTWIN_ROOT / "artifacts"
ANALYTICS_DIR = BLOODTWIN_ROOT / "analytics"
