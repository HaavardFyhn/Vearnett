from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "outputs"
WORK.mkdir(exist_ok=True)
DATABASE = ROOT / "frost_history.db"
GRID_FILE = WORK / "norway_grid_terrain.gpkg"
BASE_GRID_FILE = WORK / "norway_grid_coast_elevation.gpkg"
NORWAY_POLYGON_FILE = WORK / "norway_polygon.gpkg"


def element_update_marker(database, element):
    return Path(database).parent / "frost_element_updates" / f"{hashlib.sha256(element.encode()).hexdigest()[:16]}.stamp"

PHENOMENA = {
    "temperature": {"element": "air_temperature", "daily_agg": "mean", "min_days_per_year": 100, "min_overlap_days": 60},
    "wind": {"element": "wind_speed", "daily_agg": "mean", "min_days_per_year": 100, "min_overlap_days": 60},
    "precipitation": {"element": "sum(precipitation_amount P1D)", "daily_agg": "sum", "min_days_per_year": 100, "min_overlap_days": 60},
    "snow": {"element": "surface_snow_thickness", "daily_agg": "mean", "min_days_per_year": 100, "min_overlap_days": 60},
    "humidity": {"element": "relative_humidity", "daily_agg": "mean", "min_days_per_year": 100, "min_overlap_days": 60},
    # Minute values are reduced to one daily station mean while downloading.
    "road_temperature": {"element": "mean(road_surface_temperature PT1M)", "daily_agg": "mean", "min_days_per_year": 100,
                         "min_overlap_days": 60, "download_daily_mean": True},
}

PAIR_FEATURES = [

    "distance_m",
    "elev_diff_m",

    "coast_fjord_diff_m",
    "coast_ocean_diff_m",

    "same_objtype",

    "x_diff",
    "y_diff",

    "relief_diff_10km",
    "rugstd_diff_10km",
    "tpi_diff_10km",

    "relief_diff_25km",
    "rugstd_diff_25km",
    "tpi_diff_25km",
]


TARGET_TYPE = [ "prediction_rmse", "class_error" ]

RANDOM_STATE = 42
N_TREES = 300
MIN_SAMPLES_LEAF = 8
MAX_TRAIN_PAIRS = 500_000
TRAIN_FRACTION = 0.80
MIN_VALID_DAYS = 20
TARGET_CHUNK = 1000
CANDIDATE_BATCH = 500
TOP_N = 100
MIN_CANDIDATE_SEPARATION_M = 25_000
IMPROVEMENT_THRESHOLD = 0.01
PRECIP_WET_THRESHOLD_MM = 0.1
SNOW_PRESENT_THRESHOLD_CM = 1.0

USE_RADIUS_CUTOFF = True
THRESHOLD_RADIUS_M = 20_000

# Frost API settings
FROST_BASE_URL = "https://frost.met.no"
FROST_CLIENT_ID_ENV = "FROST_CLIENT_ID"
FROST_SOURCE_BATCH_SIZE = 48

DOWNLOAD_TIMEOUT_SECONDS = 120
DOWNLOAD_MONTHLY = True
