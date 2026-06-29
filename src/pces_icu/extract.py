from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .io import write_jsonl
from .schema import Event, PatientRecord, abnormal_flag, canonical_concept, canonical_intervention, safe_float


def _read_csv_chunks(path: Path, chunksize: int):
    """Yield all readable chunks and preserve partial data from truncated gzip files."""
    reader = pd.read_csv(path, compression="infer", chunksize=chunksize, low_memory=False)
    try:
        while True:
            try:
                yield next(reader)
            except StopIteration:
                return
            except EOFError as exc:
                print(
                    f"[extract] WARNING truncated_file={path} detail={exc}; keeping readable prefix",
                    file=sys.stderr,
                    flush=True,
                )
                return
    finally:
        reader.close()


def _dictionary(path: Path) -> dict[str, tuple[str, str]]:
    frame = pd.read_csv(path, compression="infer", low_memory=False)
    result: dict[str, tuple[str, str]] = {}
    for row in frame.itertuples(index=False):
        concept = canonical_concept(getattr(row, "label", ""))
        if concept:
            result[str(getattr(row, "itemid"))] = (concept, str(getattr(row, "unitname", "") or ""))
    return result


def _relative_minutes(times: pd.Series, intimes: pd.Series) -> pd.Series:
    return ((pd.to_datetime(times, errors="coerce") - pd.to_datetime(intimes, errors="coerce")).dt.total_seconds() / 60).round()


def _append_event(
    store: dict[str, list[Event]],
    stay_id: str,
    minute: Any,
    source: str,
    concept: str,
    raw_value: Any,
    value: Any,
    unit: Any,
    lower: Any = None,
    upper: Any = None,
    raw_flag: Any = None,
) -> None:
    if pd.isna(minute):
        return
    numeric = safe_float(value)
    event_id = f"{source[:1].upper()}{stay_id}_{len(store[stay_id]):06d}"
    store[stay_id].append(
        Event(
            event_id=event_id,
            relative_minute=int(minute),
            source=source,
            concept=concept,
            raw_value=None if pd.isna(raw_value) else raw_value,
            canonical_value=numeric if numeric is not None else (None if pd.isna(value) else str(value)),
            unit="" if pd.isna(unit) else str(unit),
            abnormal_flag=abnormal_flag(numeric, lower, upper, raw_flag),
        )
    )


def extract_mimic_like_events(
    data_root: str | Path,
    cohort: pd.DataFrame,
    output: str | Path,
    hospital_dir: str = "hosp",
    icu_dir: str = "icu",
    window_hours: int = 6,
    chunksize: int = 500_000,
) -> int:
    root = Path(data_root)
    window_minutes = window_hours * 60
    cohort = cohort.copy()
    cohort["stay_id"] = cohort["stay_id"].astype(str)
    cohort["hadm_id"] = cohort["hadm_id"].astype(str)
    cohort["intime"] = pd.to_datetime(cohort["intime"])
    stay_meta = cohort.set_index("stay_id")
    hadm_to_stay = cohort.set_index("hadm_id")[["stay_id", "intime"]]
    store: dict[str, list[Event]] = defaultdict(list)

    def progress(stage: str, chunk_index: int, rows: int) -> None:
        if chunk_index == 1 or chunk_index % 10 == 0:
            event_count = sum(len(events) for events in store.values())
            print(
                f"[extract] stage={stage} chunk={chunk_index} rows={rows} events={event_count}",
                file=sys.stderr,
                flush=True,
            )

    print("[extract] stage=chart dictionary", file=sys.stderr, flush=True)
    item_map = _dictionary(root / icu_dir / "d_items.csv.gz")
    chart_path = root / icu_dir / "chartevents.csv.gz"
    for chunk_index, chunk in enumerate(_read_csv_chunks(chart_path, chunksize), 1):
        progress("chart", chunk_index, len(chunk))
        chunk["stay_id"] = chunk["stay_id"].astype("Int64").astype(str)
        chunk = chunk[chunk["stay_id"].isin(stay_meta.index)]
        if chunk.empty:
            continue
        chunk = chunk.join(stay_meta[["intime"]], on="stay_id")
        chunk["minute"] = _relative_minutes(chunk["charttime"], chunk["intime"])
        chunk = chunk[chunk["minute"].between(0, window_minutes)]
        for row in chunk.itertuples(index=False):
            mapped = item_map.get(str(row.itemid))
            if not mapped:
                continue
            concept, default_unit = mapped
            _append_event(store, row.stay_id, row.minute, "chart", concept, row.value, row.valuenum, row.valueuom or default_unit, raw_flag=getattr(row, "warning", None))

    print("[extract] stage=lab dictionary", file=sys.stderr, flush=True)
    lab_map = _dictionary(root / hospital_dir / "d_labitems.csv.gz")
    lab_path = root / hospital_dir / "labevents.csv.gz"
    for chunk_index, chunk in enumerate(_read_csv_chunks(lab_path, chunksize), 1):
        progress("lab", chunk_index, len(chunk))
        chunk = chunk.dropna(subset=["hadm_id"]).copy()
        chunk["hadm_id"] = chunk["hadm_id"].astype("Int64").astype(str)
        chunk = chunk[chunk["hadm_id"].isin(hadm_to_stay.index)]
        if chunk.empty:
            continue
        chunk = chunk.join(hadm_to_stay, on="hadm_id")
        chunk["minute"] = _relative_minutes(chunk["charttime"], chunk["intime"])
        chunk = chunk[chunk["minute"].between(0, window_minutes)]
        for row in chunk.itertuples(index=False):
            mapped = lab_map.get(str(row.itemid))
            if not mapped:
                continue
            concept, default_unit = mapped
            _append_event(store, row.stay_id, row.minute, "lab", concept, row.value, row.valuenum, row.valueuom or default_unit, row.ref_range_lower, row.ref_range_upper, row.flag)

    input_path = root / icu_dir / "inputevents.csv.gz"
    if input_path.exists():
        labels = pd.read_csv(root / icu_dir / "d_items.csv.gz", compression="infer", low_memory=False)
        intervention_map = {str(row.itemid): canonical_intervention(row.label) for row in labels.itertuples(index=False)}
        intervention_map = {key: value for key, value in intervention_map.items() if value}
        print("[extract] stage=input", file=sys.stderr, flush=True)
        for chunk_index, chunk in enumerate(_read_csv_chunks(input_path, chunksize), 1):
            progress("input", chunk_index, len(chunk))
            chunk["stay_id"] = chunk["stay_id"].astype("Int64").astype(str)
            chunk = chunk[chunk["stay_id"].isin(stay_meta.index)]
            if chunk.empty:
                continue
            chunk = chunk.join(stay_meta[["intime"]], on="stay_id")
            chunk["minute"] = _relative_minutes(chunk["starttime"], chunk["intime"])
            chunk = chunk[chunk["minute"].between(0, window_minutes)]
            for row in chunk.itertuples(index=False):
                concept = intervention_map.get(str(row.itemid))
                if concept:
                    value = getattr(row, "rate", None) if pd.notna(getattr(row, "rate", None)) else getattr(row, "amount", 1.0)
                    unit = getattr(row, "rateuom", "") if pd.notna(getattr(row, "rate", None)) else getattr(row, "amountuom", "")
                    _append_event(store, row.stay_id, row.minute, "medication_admin", concept, value, value, unit, raw_flag="abnormal")
    else:
        prescriptions = root / hospital_dir / "prescriptions.csv.gz"
        if prescriptions.exists():
            print("[extract] stage=prescriptions", file=sys.stderr, flush=True)
            for chunk_index, chunk in enumerate(_read_csv_chunks(prescriptions, chunksize), 1):
                progress("prescriptions", chunk_index, len(chunk))
                chunk = chunk.dropna(subset=["hadm_id"]); chunk["hadm_id"] = chunk["hadm_id"].astype("Int64").astype(str)
                chunk = chunk[chunk["hadm_id"].isin(hadm_to_stay.index)]
                if chunk.empty:
                    continue
                chunk = chunk.join(hadm_to_stay, on="hadm_id"); chunk["minute"] = _relative_minutes(chunk["starttime"], chunk["intime"])
                chunk = chunk[chunk["minute"].between(0, window_minutes)]
                for row in chunk.itertuples(index=False):
                    concept = canonical_intervention(getattr(row, "drug", ""))
                    if concept:
                        value = getattr(row, "dose_val_rx", 1.0); unit = getattr(row, "dose_unit_rx", "")
                        _append_event(store, row.stay_id, row.minute, "medication_order", concept, getattr(row, "drug", ""), value, unit, raw_flag="abnormal")

    procedure_path = root / icu_dir / "procedureevents.csv.gz"
    if procedure_path.exists():
        labels = pd.read_csv(root / icu_dir / "d_items.csv.gz", compression="infer", low_memory=False)
        procedure_map = {str(row.itemid): canonical_intervention(row.label) for row in labels.itertuples(index=False)}
        procedure_map = {key: value for key, value in procedure_map.items() if value}
        print("[extract] stage=procedure", file=sys.stderr, flush=True)
        for chunk_index, chunk in enumerate(_read_csv_chunks(procedure_path, chunksize), 1):
            progress("procedure", chunk_index, len(chunk))
            chunk["stay_id"] = chunk["stay_id"].astype("Int64").astype(str); chunk = chunk[chunk["stay_id"].isin(stay_meta.index)]
            if chunk.empty:
                continue
            chunk = chunk.join(stay_meta[["intime"]], on="stay_id"); chunk["minute"] = _relative_minutes(chunk["starttime"], chunk["intime"])
            chunk = chunk[chunk["minute"].between(0, window_minutes)]
            for row in chunk.itertuples(index=False):
                concept = procedure_map.get(str(row.itemid))
                if concept:
                    _append_event(store, row.stay_id, row.minute, "procedure", concept, concept, 1.0, "", raw_flag="abnormal")

    def records() -> Iterable[dict[str, Any]]:
        for row in cohort.itertuples(index=False):
            record = PatientRecord(
                patient_id=str(row.patient_id),
                stay_id=str(row.stay_id),
                label=int(row.label),
                dataset=str(row.dataset),
                hospital_id=str(row.hospital_id),
                split=str(getattr(row, "split", "")),
                events=sorted(store.get(str(row.stay_id), []), key=lambda event: (event.relative_minute, event.event_id)),
            )
            yield record.to_dict()
    return write_jsonl(output, records())


EICU_VITAL_COLUMNS = {
    "temperature": "temperature",
    "sao2": "spo2",
    "heartrate": "heart_rate",
    "respiration": "respiratory_rate",
    "systemicsystolic": "sbp",
    "systemicdiastolic": "dbp",
    "systemicmean": "map",
}


def extract_eicu_events(
    data_root: str | Path,
    cohort: pd.DataFrame,
    output: str | Path,
    window_hours: int = 6,
    chunksize: int = 500_000,
) -> int:
    root = Path(data_root)
    window_minutes = window_hours * 60
    cohort = cohort.copy()
    cohort["stay_id"] = cohort["stay_id"].astype(str)
    stay_ids_numeric = pd.to_numeric(cohort["stay_id"], errors="raise").to_numpy(dtype=np.int64)
    store: dict[str, list[Event]] = defaultdict(list)

    def progress(stage: str, chunk_index: int, rows: int) -> None:
        if chunk_index == 1 or chunk_index % 10 == 0:
            event_count = sum(len(events) for events in store.values())
            print(
                f"[extract] stage={stage} chunk={chunk_index} rows={rows} events={event_count}",
                file=sys.stderr,
                flush=True,
            )

    for filename in ("vitalPeriodic.csv.gz", "vitalAperiodic.csv.gz"):
        path = root / filename
        stage = path.stem.split(".")[0]
        print(f"[extract] stage={stage}", file=sys.stderr, flush=True)
        for chunk_index, chunk in enumerate(_read_csv_chunks(path, chunksize), 1):
            progress(stage, chunk_index, len(chunk))
            mask = np.isin(chunk["patientunitstayid"].to_numpy(dtype=np.int64), stay_ids_numeric)
            chunk = chunk.loc[mask].copy()
            chunk["stay_id"] = chunk["patientunitstayid"].astype("int64").astype(str)
            offset_col = "observationoffset"
            chunk = chunk[pd.to_numeric(chunk[offset_col], errors="coerce").between(0, window_minutes)]
            mappings = dict(EICU_VITAL_COLUMNS)
            mappings.update({"noninvasivesystolic": "sbp", "noninvasivediastolic": "dbp", "noninvasivemean": "map"})
            for row in chunk.itertuples(index=False):
                minute = getattr(row, offset_col)
                for column, concept in mappings.items():
                    if hasattr(row, column):
                        value = getattr(row, column)
                        if pd.notna(value):
                            _append_event(store, row.stay_id, minute, "vital", concept, value, value, "")
    print("[extract] stage=lab", file=sys.stderr, flush=True)
    for chunk_index, chunk in enumerate(_read_csv_chunks(root / "lab.csv.gz", chunksize), 1):
        progress("lab", chunk_index, len(chunk))
        mask = np.isin(chunk["patientunitstayid"].to_numpy(dtype=np.int64), stay_ids_numeric)
        chunk = chunk.loc[mask].copy()
        chunk["stay_id"] = chunk["patientunitstayid"].astype("int64").astype(str)
        chunk = chunk[pd.to_numeric(chunk["labresultoffset"], errors="coerce").between(0, window_minutes)]
        for row in chunk.itertuples(index=False):
            concept = canonical_concept(row.labname)
            if concept:
                _append_event(store, row.stay_id, row.labresultoffset, "lab", concept, row.labresulttext, row.labresult, row.labmeasurenamesystem)

    print("[extract] stage=medication", file=sys.stderr, flush=True)
    for chunk_index, chunk in enumerate(_read_csv_chunks(root / "medication.csv.gz", chunksize), 1):
        progress("medication", chunk_index, len(chunk))
        mask = np.isin(chunk["patientunitstayid"].to_numpy(dtype=np.int64), stay_ids_numeric)
        chunk = chunk.loc[mask].copy()
        chunk["stay_id"] = chunk["patientunitstayid"].astype("int64").astype(str)
        chunk = chunk[pd.to_numeric(chunk["drugstartoffset"], errors="coerce").between(0, window_minutes)]
        for row in chunk.itertuples(index=False):
            concept = canonical_intervention(getattr(row, "drugname", ""))
            if concept:
                _append_event(store, row.stay_id, row.drugstartoffset, "medication_order", concept, row.drugname, getattr(row, "dosage", 1.0), getattr(row, "routeadmin", ""), raw_flag="abnormal")

    treatment_path = root / "treatment.csv.gz"
    if treatment_path.exists():
        print("[extract] stage=treatment", file=sys.stderr, flush=True)
        for chunk_index, chunk in enumerate(_read_csv_chunks(treatment_path, chunksize), 1):
            progress("treatment", chunk_index, len(chunk))
            mask = np.isin(chunk["patientunitstayid"].to_numpy(dtype=np.int64), stay_ids_numeric)
            chunk = chunk.loc[mask].copy()
            chunk["stay_id"] = chunk["patientunitstayid"].astype("int64").astype(str)
            chunk = chunk[pd.to_numeric(chunk["treatmentoffset"], errors="coerce").between(0, window_minutes)]
            for row in chunk.itertuples(index=False):
                concept = canonical_intervention(getattr(row, "treatmentstring", ""))
                if concept:
                    _append_event(store, row.stay_id, row.treatmentoffset, "treatment", concept, row.treatmentstring, 1.0, "", raw_flag="abnormal")

    def records() -> Iterable[dict[str, Any]]:
        for row in cohort.itertuples(index=False):
            yield PatientRecord(
                patient_id=str(row.patient_id), stay_id=str(row.stay_id), label=int(row.label), dataset="eicu",
                hospital_id=str(row.hospital_id), split=str(getattr(row, "split", "")),
                events=sorted(store.get(str(row.stay_id), []), key=lambda event: (event.relative_minute, event.event_id)),
            ).to_dict()
    return write_jsonl(output, records())
