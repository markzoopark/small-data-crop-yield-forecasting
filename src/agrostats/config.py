"""Configuration constants for agrostats project."""

from __future__ import annotations

from pathlib import Path

# Path constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# Random seeds for reproducibility
RANDOM_STATE = 42
RANDOM_SEED = 42

# Data constants
MIN_YEAR = 2010
MAX_YEAR = 2024
TARGET_CROPS = ("Пшениця", "Кукурудза", "Соняшник")

# Train/validation/test split
SPLIT_TRAIN_END = 2018
SPLIT_VAL_END = 2021
TEST_START = 2022
