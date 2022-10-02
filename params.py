"""
taxifare model package params
load and validate the environment variables in the `.env`
"""

import os
import numpy as np

DATASET_SIZE = "10k"            # ["1k","10k", "100k", "500k"]
VALIDATION_DATASET_SIZE = "10k" # ["1k", "10k", "100k", "500k"]
CHUNK_SIZE = 5000
LOCAL_DATA_PATH = "data"
LOCAL_REGISTRY_PATH = "training_outputs"

# Use this to optimize loading of raw_data with headers: pd.read_csv(..., dtypes=..., headers=True)
DTYPES_RAW_OPTIMIZED = {
    "key": "O",
    "fare_amount": "float32",
    "pickup_datetime": "O",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8"
}

COLUMN_NAMES_RAW = DTYPES_RAW_OPTIMIZED.keys()

# Use this to optimize loading of raw_data without headers: pd.read_csv(..., dtypes=..., headers=False)
DTYPES_RAW_OPTIMIZED_HEADLESS = {
    0: "O",
    1: "float32",
    2: "O",
    3: "float32",
    4: "float32",
    5: "float32",
    6: "float32",
    7: "int8"
}

DTYPES_PROCESSED_OPTIMIZED = np.float32



################## VALIDATIONS #################

env_valid_options = dict(
    DATASET_SIZE=["1k", "10k", "100k", "500k", "50M", "new"],
    VALIDATION_DATASET_SIZE=["1k", "10k", "100k", "500k", "500k", "new"],
    DATA_SOURCE=["local", "big query"],
    MODEL_TARGET=["local", "gcs", "mlflow"],)
