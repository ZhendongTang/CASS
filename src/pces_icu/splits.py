from __future__ import annotations

import hashlib

import pandas as pd


def _hash_fraction(value: str, seed: int) -> float:
    digest = hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") / float(2**64)


def assign_patient_splits(frame: pd.DataFrame, seed: int = 2026, train: float = 0.7, val: float = 0.1) -> pd.DataFrame:
    if train <= 0 or val < 0 or train + val >= 1:
        raise ValueError("Expected 0 < train and train + val < 1")
    result = frame.copy()
    fractions = result["patient_id"].astype(str).map(lambda value: _hash_fraction(value, seed))
    result["split"] = "test"
    result.loc[fractions < train + val, "split"] = "val"
    result.loc[fractions < train, "split"] = "train"
    return result


def assign_hospital_splits(frame: pd.DataFrame, seed: int = 2026, train: float = 0.7, val: float = 0.1) -> pd.DataFrame:
    if frame["hospital_id"].astype(str).eq("").any():
        raise ValueError("hospital_id is required for hospital-held-out splits")
    hospitals = pd.DataFrame({"hospital_id": sorted(frame["hospital_id"].astype(str).unique())})
    hospitals["fraction"] = hospitals["hospital_id"].map(lambda value: _hash_fraction(value, seed))
    hospitals["split"] = "test"
    hospitals.loc[hospitals["fraction"] < train + val, "split"] = "val"
    hospitals.loc[hospitals["fraction"] < train, "split"] = "train"
    return frame.drop(columns=["split"], errors="ignore").merge(hospitals[["hospital_id", "split"]], on="hospital_id", how="left", validate="many_to_one")


def assert_disjoint(frame: pd.DataFrame, key: str = "patient_id") -> None:
    memberships = frame.groupby(key)["split"].nunique()
    bad = memberships[memberships > 1]
    if len(bad):
        raise AssertionError(f"{len(bad)} {key} values occur in multiple splits")

