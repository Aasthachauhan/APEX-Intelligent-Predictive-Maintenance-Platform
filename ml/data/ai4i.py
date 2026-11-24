"""AI4I-2020-style data: real-file loader + SIMULATOR.

The real UCI AI4I 2020 file (https://archive.ics.uci.edu/dataset/601) is itself synthetic.
`simulate()` is NOT that dataset. It re-creates the failure rules documented on the UCI page
(TWF, HDF, PWF, OSF, RNF) with approximate sensor distributions, so the pipeline can be
developed/tested offline. Metrics on simulated data MUST NOT be reported as AI4I results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RAW_COLS = {
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Type": "type",
    "Machine failure": "machine_failure",
}
FEATURES_NUM = ["air_temp", "process_temp", "rpm", "torque", "tool_wear"]
FEATURES_CAT = ["type"]
TARGET = "machine_failure"
MODE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
OSF_LIMIT = {"L": 11000, "M": 12000, "H": 13000}  # min*Nm, from UCI documentation


def load_real(path: str) -> pd.DataFrame:
    return from_frame(pd.read_csv(path))


def from_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Accept raw UCI column names or APEX internal names; reject anything else."""
    internal = FEATURES_NUM + FEATURES_CAT + [TARGET]
    if all(c in df.columns for c in internal):
        return df[internal + [m for m in MODE_COLS if m in df.columns]]
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Not an AI4I file, missing columns: {missing}")
    out = df.rename(columns=RAW_COLS)
    return out[list(RAW_COLS.values()) + [m for m in MODE_COLS if m in out]]


def simulate(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = rng.choice(["L", "M", "H"], n, p=[0.6, 0.3, 0.1])
    air = 300 + 2 * rng.standard_normal(n)
    proc = air + 10 + rng.normal(0, 1.0, n)
    rpm = np.round(1500 + 180 * rng.standard_normal(n) + rng.exponential(60, n))
    # real AI4I shows torque ~ inversely related to rpm (power roughly constant): derive torque from power
    power0 = rng.normal(6300, 1050, n)
    torque = np.clip(power0 / (rpm * 2 * np.pi / 60), 3, None).round(1)
    wear = np.clip(rng.integers(0, 200, n) + rng.integers(0, 40, n), 0, 253)
    power = torque * rpm * 2 * np.pi / 60
    twf = (wear >= 200) & (rng.random(n) < 0.12)
    hdf = ((proc - air) < 8.6) & (rpm < 1380)
    pwf = (power < 3500) | (power > 9000)
    osf = wear * torque > np.vectorize(OSF_LIMIT.get)(t)
    rnf = rng.random(n) < 0.001
    fail = twf | hdf | pwf | osf | rnf
    return pd.DataFrame(
        {
            "air_temp": air.round(1),
            "process_temp": proc.round(1),
            "rpm": rpm.astype(int),
            "torque": torque,
            "tool_wear": wear.astype(int),
            "type": t,
            "machine_failure": fail.astype(int),
            "TWF": twf.astype(int),
            "HDF": hdf.astype(int),
            "PWF": pwf.astype(int),
            "OSF": osf.astype(int),
            "RNF": rnf.astype(int),
        }
    )
