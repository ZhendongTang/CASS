from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, compression="infer", low_memory=False)


def _adult_age(anchor_age: pd.Series, anchor_year: pd.Series, intime: pd.Series) -> pd.Series:
    return pd.to_numeric(anchor_age, errors="coerce") + intime.dt.year - pd.to_numeric(anchor_year, errors="coerce")


def build_mimic_like_cohort(
    root: str | Path,
    dataset: str,
    window_hours: int = 6,
    hospital_dir: str = "hosp",
    icu_dir: str = "icu",
) -> pd.DataFrame:
    root = Path(root)
    patients = _read(root / hospital_dir / "patients.csv.gz")
    admissions = _read(root / hospital_dir / "admissions.csv.gz")
    stays = _read(root / icu_dir / "icustays.csv.gz")
    stays["intime"] = pd.to_datetime(stays["intime"], errors="coerce")
    stays["outtime"] = pd.to_datetime(stays["outtime"], errors="coerce")
    stays = stays.sort_values(["subject_id", "intime"]).drop_duplicates("subject_id", keep="first")
    frame = stays.merge(patients, on="subject_id", how="inner", validate="many_to_one")
    admission_columns = ["hadm_id", "hospital_expire_flag"]
    if dataset == "nwicu":
        admission_columns.append("discharge_location")
    frame = frame.merge(admissions[admission_columns], on="hadm_id", how="inner", validate="many_to_one")
    frame["age"] = _adult_age(frame["anchor_age"], frame["anchor_year"], frame["intime"])
    frame["duration_hours"] = (frame["outtime"] - frame["intime"]).dt.total_seconds() / 3600
    frame = frame[(frame["age"] >= 18) & (frame["duration_hours"] > window_hours)].copy()
    if dataset == "nwicu":
        # In NWICU v0.1.0, hospital_expire_flag marks only a small subset of
        # records whose discharge_location is EXPIRED.  The latter agrees with
        # in-stay death timestamps and is the auditable in-hospital endpoint.
        frame["label"] = (
            frame["discharge_location"].fillna("").astype(str).str.strip().str.upper().str.startswith("EXPIRED").astype(int)
        )
    else:
        frame["label"] = pd.to_numeric(frame["hospital_expire_flag"], errors="raise").astype(int)
    frame["dataset"] = dataset
    frame["patient_id"] = frame["subject_id"].astype(str)
    frame["stay_id"] = frame["stay_id"].astype(str)
    frame["hospital_id"] = ""
    return frame[
        [
            "patient_id",
            "subject_id",
            "hadm_id",
            "stay_id",
            "intime",
            "outtime",
            "age",
            "duration_hours",
            "label",
            "dataset",
            "hospital_id",
        ]
    ]


def _parse_eicu_age(value: object) -> float | None:
    if value == "> 89":
        return 90.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_eicu_cohort(root: str | Path, window_hours: int = 6) -> pd.DataFrame:
    root = Path(root)
    frame = _read(root / "patient.csv.gz")
    frame["unitvisitnumber"] = pd.to_numeric(frame["unitvisitnumber"], errors="coerce").fillna(999)
    unique = frame.get("uniquepid", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    frame["patient_key"] = unique.where(unique.ne(""), "hs_" + frame["patienthealthsystemstayid"].astype(str))
    frame["hospitaldischargeyear"] = pd.to_numeric(frame.get("hospitaldischargeyear"), errors="coerce").fillna(9999)
    frame = frame.sort_values(["patient_key", "hospitaldischargeyear", "patienthealthsystemstayid", "unitvisitnumber"]).drop_duplicates(
        "patient_key", keep="first"
    )
    frame["age_numeric"] = frame["age"].map(_parse_eicu_age)
    frame["duration_hours"] = pd.to_numeric(frame["unitdischargeoffset"], errors="coerce") / 60
    frame = frame[(frame["age_numeric"] >= 18) & (frame["duration_hours"] > window_hours)].copy()
    frame["label"] = frame["hospitaldischargestatus"].fillna("").str.lower().eq("expired").astype(int)
    frame["dataset"] = "eicu"
    frame["patient_id"] = frame["patient_key"].astype(str)
    frame["stay_id"] = frame["patientunitstayid"].astype(str)
    frame["hospital_id"] = frame["hospitalid"].astype(str)
    frame["intime"] = pd.NaT
    return frame[
        [
            "patient_id",
            "patienthealthsystemstayid",
            "patientunitstayid",
            "stay_id",
            "intime",
            "age_numeric",
            "duration_hours",
            "label",
            "dataset",
            "hospital_id",
        ]
    ].rename(columns={"age_numeric": "age"})


def cohort_summary(frame: pd.DataFrame) -> dict[str, float | int | str]:
    labels = pd.to_numeric(frame["label"], errors="raise")
    return {
        "dataset": str(frame["dataset"].iloc[0]) if len(frame) else "unknown",
        "n": int(len(frame)),
        "positives": int(labels.sum()),
        "positive_rate": float(labels.mean()) if len(frame) else 0.0,
        "patients": int(frame["patient_id"].nunique()),
        "hospitals": int(frame["hospital_id"].replace("", pd.NA).nunique()) if "hospital_id" in frame else 0,
    }
