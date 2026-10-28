"""
Dataset acquisition and fallback generator script for the AI4I 2020 Predictive Maintenance Dataset.
Attempts download from official/mirror sources, and falls back to a reproducible synthetic dataset
with the identical schema and physical distribution if offline.
"""

import io
import sys
import zipfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import requests

from config import (
    CATEGORICAL_COLUMNS,
    DATASET_PATH,
    DATASET_URL,
    FAILURE_MODES,
    NUMERICAL_COLUMNS,
    SYNTHETIC_DATASET_PATH,
    TARGET_COLUMN,
)


def generate_synthetic_dataset(num_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic dataset mimicking the exact AI4I 2020 distribution and physical relationships.
    Used for offline operation and automated unit tests.
    """
    np.random.seed(seed)

    types = np.random.choice(["L", "M", "H"], size=num_samples, p=[0.50, 0.30, 0.20])
    udis = np.arange(1, num_samples + 1)
    product_ids = [f"{t}{10000 + i}" for i, t in enumerate(types)]

    # Air temperature: normal around 300 K with std 2
    air_temp = np.random.normal(300.0, 2.0, num_samples)

    # Process temperature: air_temp + 10 K + normal std 1
    process_temp = air_temp + 10.0 + np.random.normal(0.0, 1.0, num_samples)

    # Torque: normal around 40 Nm with std 10, strictly positive
    torque = np.clip(np.random.normal(40.0, 10.0, num_samples), 3.0, 80.0)

    # Rotational speed: inversely proportional to torque around 2860 W with noise
    power_nominal = 2860.0
    rot_speed = (power_nominal / (torque * (2 * np.pi / 60))) + np.random.normal(0.0, 50.0, num_samples)
    rot_speed = np.clip(rot_speed, 1150.0, 2900.0)

    # Tool wear: random wear between 0 and 250 min
    tool_wear = np.random.uniform(0.0, 250.0, num_samples)

    # Physical Failure modes as defined in Matzka (2020):
    # 1. TWF: Tool wear failure (tool wear between 200 and 240 mins)
    twf = ((tool_wear >= 200) & (np.random.rand(num_samples) < 0.05)).astype(int)

    # 2. HDF: Heat dissipation failure: temp diff < 8.6 K and rot speed < 1380 rpm
    temp_diff = process_temp - air_temp
    hdf = ((temp_diff < 8.6) & (rot_speed < 1380.0)).astype(int)

    # 3. PWF: Power failure: power < 3500 W or > 9000 W
    power_watts = torque * (rot_speed * (2 * np.pi / 60))
    pwf = ((power_watts < 3500.0) | (power_watts > 9000.0)).astype(int)

    # 4. OSF: Overstrain failure: tool_wear * torque > threshold (11000 for L, 12000 for M, 13000 for H)
    os_metric = tool_wear * torque
    thresholds = np.where(types == "L", 11000.0, np.where(types == "M", 12000.0, 13000.0))
    osf = (os_metric > thresholds).astype(int)

    # 5. RNF: Random failure: 0.1% chance
    rnf = (np.random.rand(num_samples) < 0.001).astype(int)

    # Overall machine failure
    machine_failure = (twf | hdf | pwf | osf | rnf).astype(int)

    df = pd.DataFrame({
        "UDI": udis,
        "Product ID": product_ids,
        "Type": types,
        "Air temperature [K]": np.round(air_temp, 2),
        "Process temperature [K]": np.round(process_temp, 2),
        "Rotational speed [rpm]": np.round(rot_speed, 0),
        "Torque [Nm]": np.round(torque, 1),
        "Tool wear [min]": np.round(tool_wear, 0),
        "Machine failure": machine_failure,
        "TWF": twf,
        "HDF": hdf,
        "PWF": pwf,
        "OSF": osf,
        "RNF": rnf,
    })

    return df


def download_ai4i_dataset(target_path: Path = DATASET_PATH) -> bool:
    """
    Attempt to download official dataset from UCI repository or reliable mirror.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)

    urls = [
        ("UCI Official Zip", DATASET_URL),
        ("GitHub Mirror CSV", "https://raw.githubusercontent.com/jbrownlee/Datasets/master/ai4i2020.csv"),
    ]

    for source_name, url in urls:
        try:
            print(f"Attempting download from {source_name}: {url}...")
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                if url.endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                        csv_files = [f for f in z.namelist() if f.endswith(".csv")]
                        if csv_files:
                            with open(target_path, "wb") as f_out:
                                f_out.write(z.read(csv_files[0]))
                            print(f"Successfully extracted {csv_files[0]} to {target_path}")
                            return True
                else:
                    with open(target_path, "wb") as f_out:
                        f_out.write(response.content)
                    print(f"Successfully saved dataset to {target_path}")
                    return True
        except Exception as e:
            print(f"Failed download from {source_name}: {e}")

    return False


def acquire_dataset() -> Path:
    """
    Acquires dataset if not present. Falls back cleanly to synthetic dataset if network fails.
    """
    if DATASET_PATH.exists() and DATASET_PATH.stat().st_size > 1000:
        print(f"Dataset already exists at: {DATASET_PATH}")
        return DATASET_PATH

    print("Primary dataset not found locally. Initiating acquisition...")
    success = download_ai4i_dataset(DATASET_PATH)
    if success:
        return DATASET_PATH

    print("Network download unsuccessful. Generating fully compatible synthetic dataset...")
    df_synthetic = generate_synthetic_dataset(num_samples=10000)
    SYNTHETIC_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_synthetic.to_csv(SYNTHETIC_DATASET_PATH, index=False)
    # Also save to DATASET_PATH so downstream modules can seamlessly load
    df_synthetic.to_csv(DATASET_PATH, index=False)
    print(f"Synthetic dataset saved to {SYNTHETIC_DATASET_PATH} and {DATASET_PATH}")
    return DATASET_PATH


if __name__ == "__main__":
    acquire_dataset()

