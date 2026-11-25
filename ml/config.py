import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Thêm project root vào sys.path nếu chưa có
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Data directories
DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
RAW_RETAIL_FILE  = RAW_DATA_DIR / "online_retail_II.xlsx"

CLEAN_RETAIL_FILE = (
    PROCESSED_DATA_DIR
    / "online_retail_clean.csv"
)

VALID_SALES_FILE = (
    PROCESSED_DATA_DIR
    / "valid_sales.csv"
)

RETURNS_FILE = (
    PROCESSED_DATA_DIR
    / "returns_and_cancellations.csv"
)

INVALID_TRANSACTIONS_FILE = (
    PROCESSED_DATA_DIR
    / "invalid_transactions.csv"
)

DAILY_REVENUE_FILE = (
    PROCESSED_DATA_DIR
    / "daily_revenue.csv"
)

# Artifact directories
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODELS_DIR = ARTIFACTS_DIR / "models"
FORECASTS_DIR = ARTIFACTS_DIR / "forecasts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

# Project directories

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
ANALYTICS_DIR = (
    ARTIFACTS_DIR
    / "analytics"
)

TIME_SERIES_DIR = (
    ARTIFACTS_DIR
    / "time_series"
)
# Directory initialization

DIRECTORIES = [
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    MODELS_DIR,
    FORECASTS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    ANALYTICS_DIR,
    TIME_SERIES_DIR,
]


def create_directories() -> None:
    """
    Create required project directories if they do not already exist.
    """

    for directory in DIRECTORIES:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )
    # if not RAW_SALES_FILE.exists():
    #     RAW_SALES_FILE.write_text("date,amount\n2024-01-01,100\n")
    