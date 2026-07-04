from __future__ import annotations

import gzip
import sys
import tempfile
import unittest
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pces_icu.cohort import build_eicu_cohort, build_mimic_like_cohort
from pces_icu.metrics import auroc, average_precision, binary_metrics, permutation_metrics
from pces_icu.io import read_jsonl, write_json, write_jsonl
from pces_icu.schema import Event, PatientRecord, canonical_concept, canonical_intervention, validate_events
from pces_icu.serialization_pipeline import build_serialization_pipeline
from pces_icu.serialize import (
    ClinTimeCompactConfig,
    ClinTimeJSONConfig,
    ClinTimeXMLConfig,
    SerializationConfig,
    extract_event_ids,
    paired_views,
    serialize_clintime_compact,
    serialize_clintime_json,
    serialize_clintime_xml,
    serialize_events,
)
from pces_icu.splits import assert_disjoint, assign_hospital_splits, assign_patient_splits
from pces_icu.synthetic import make_synthetic_records
from pces_icu.tasks import build_future_intervention_task, build_los_binary_task


class SerializerTests(unittest.TestCase):
    def setUp(self):
        self.events = [
            Event("E1", 10, "vital", "heart_rate", 120, 120.0, "bpm", "high"),
            Event("E2", 10, "vital", "sbp", 85, 85.0, "mmHg", "low"),
            Event("E3", 20, "lab", "lactate", 4.2, 4.2, "mmol/L", "high"),
        ]

    def test_views_have_identical_events_and_time_order(self):
        a, b = paired_views(self.events, SerializationConfig(), seed=2)
        self.assertEqual(sorted(extract_event_ids(a)), sorted(extract_event_ids(b)))
        self.assertLess(a.index("<T=+10m>"), a.index("<T=+20m>"))
        self.assertLess(b.index("<T=+10m>"), b.index("<T=+20m>"))

    def test_negative_control_can_shuffle_time(self):
        text = serialize_events(self.events, SerializationConfig(shuffle_across_timestamps=True), seed=1)
        self.assertEqual(sorted(extract_event_ids(text)), ["E1", "E2", "E3"])

    def test_flat_serialization_removes_set_markers(self):
        text = serialize_events(self.events, SerializationConfig(include_set_markers=False, permute_within_sets=False), seed=1)
        self.assertNotIn("<SET>", text)
        self.assertNotIn("</SET>", text)
        self.assertEqual(sorted(extract_event_ids(text)), ["E1", "E2", "E3"])

    def test_matched_budget(self):
        a, b = paired_views(self.events * 10, SerializationConfig(max_events=2), seed=4)
        self.assertEqual(sorted(extract_event_ids(a)), sorted(extract_event_ids(b)))
        self.assertEqual(len(extract_event_ids(a)), 2)

    def test_clintime_xml_is_valid_and_traceable(self):
        events = self.events + [Event("E4", 50, "vital", "heart_rate", 130, 130.0, "bpm", "high")]
        text = serialize_clintime_xml(events, ClinTimeXMLConfig(), seed=7)
        root = ET.fromstring(text)
        self.assertEqual(root.tag, "icu_stream")
        blocks = root.findall("timeblock")
        self.assertTrue(blocks)
        self.assertTrue(all(block.attrib["semantics"] == "unordered" for block in blocks))
        heart_rate = next(node for node in root.iter("observation") if node.attrib["concept"] == "heart_rate")
        self.assertEqual(heart_rate.attrib["first"], "120")
        self.assertEqual(heart_rate.attrib["last"], "130")
        self.assertEqual(heart_rate.attrib["trend"], "up")
        self.assertEqual(sorted(extract_event_ids(text)), ["E1", "E2", "E3", "E4"])

    def test_clintime_permutations_keep_same_evidence(self):
        config = ClinTimeXMLConfig(max_chars=2000)
        a = serialize_clintime_xml(self.events, config, seed=1)
        b = serialize_clintime_xml(self.events, config, seed=2)
        self.assertEqual(sorted(extract_event_ids(a)), sorted(extract_event_ids(b)))
        self.assertLessEqual(len(a), config.max_chars)
        self.assertLessEqual(len(b), config.max_chars)

    def test_clintime_canonical_is_invariant_to_input_order(self):
        config = ClinTimeXMLConfig(permute_within_blocks=False)
        forward = serialize_clintime_xml(self.events, config, seed=1)
        reverse = serialize_clintime_xml(list(reversed(self.events)), config, seed=999)
        self.assertEqual(forward, reverse)

    def test_compact_clintime_training_form_is_shorter_than_xml(self):
        events = self.events + [Event("E4", 50, "vital", "heart_rate", 130, 130.0, "bpm", "high")]
        config = ClinTimeCompactConfig(permute_within_blocks=False)
        compact = serialize_clintime_compact(events, config, seed=7)
        xml = serialize_clintime_xml(events, ClinTimeXMLConfig(permute_within_blocks=False), seed=7)
        self.assertNotIn("ClinTime-Compact/1", compact)
        self.assertIn("HR", compact)
        self.assertNotIn("ids=", compact)
        self.assertLess(len(compact), len(xml))

    def test_compact_trace_form_keeps_same_evidence(self):
        config = ClinTimeCompactConfig(max_chars=2000, include_evidence_ids=True, include_legend=True)
        a = serialize_clintime_compact(self.events, config, seed=1)
        b = serialize_clintime_compact(self.events, config, seed=2)
        self.assertIn("ClinTime-Compact/1", a)
        self.assertEqual(sorted(extract_event_ids(a)), sorted(extract_event_ids(b)))
        self.assertLessEqual(len(a), config.max_chars)
        self.assertLessEqual(len(b), config.max_chars)

    def test_clintime_json_training_form_is_valid_and_short(self):
        events = self.events + [Event("E4", 50, "vital", "heart_rate", 130, 130.0, "bpm", "high")]
        text = serialize_clintime_json(events, ClinTimeJSONConfig(permute_within_blocks=False), seed=7)
        parsed = json.loads(text)
        self.assertEqual(parsed["fmt"], "CTJ1")
        self.assertIn('"c":"HR"', text)
        self.assertNotIn('"ids"', text)
        self.assertLess(len(text), len(serialize_clintime_xml(events, ClinTimeXMLConfig(permute_within_blocks=False), seed=7)))

    def test_clintime_json_trace_extracts_evidence(self):
        config = ClinTimeJSONConfig(include_evidence_ids=True)
        a = serialize_clintime_json(self.events, config, seed=1)
        b = serialize_clintime_json(self.events, config, seed=2)
        self.assertEqual(sorted(extract_event_ids(a)), sorted(extract_event_ids(b)))

    def test_serialization_pipeline_writes_all_public_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                PatientRecord("p1", "s1", 1, "mimic", split="train", events=self.events),
                PatientRecord("p2", "s2", 0, "mimic", split="test", events=self.events[:1]),
            ]
            events_path = root / "events.jsonl.gz"
            write_jsonl(events_path, (record.to_dict() for record in records))
            manifest = build_serialization_pipeline(events_path, root / "serialized")
            self.assertEqual(manifest["formats"], ["cass", "json", "xml", "meds"])
            for fmt in manifest["formats"]:
                rows = list(read_jsonl(root / "serialized" / f"{fmt}.jsonl.gz"))
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]["format"], fmt)
                self.assertIn("text", rows[0])
            self.assertIn("ClinTime-FeatureSummary/1", list(read_jsonl(root / "serialized" / "cass.jsonl.gz"))[0]["text"])
            self.assertIn('"fmt":"CTJ1"', list(read_jsonl(root / "serialized" / "json.jsonl.gz"))[0]["text"])
            self.assertIn("<icu_stream", list(read_jsonl(root / "serialized" / "xml.jsonl.gz"))[0]["text"])
            self.assertIn("MEDS-like event table", list(read_jsonl(root / "serialized" / "meds.jsonl.gz"))[0]["text"])


class MetricTests(unittest.TestCase):
    def test_perfect_metrics(self):
        y = [0, 0, 1, 1]
        p = [0.1, 0.2, 0.8, 0.9]
        self.assertAlmostEqual(auroc(y, p), 1.0)
        self.assertAlmostEqual(average_precision(y, p), 1.0)
        self.assertLess(binary_metrics(y, p)["brier"], 0.05)

    def test_permutation_metrics(self):
        result = permutation_metrics(np.asarray([[0.1, 0.1], [0.9, 0.9]]))
        self.assertEqual(result["label_flip_rate"], 0.0)
        self.assertAlmostEqual(result["mean_prediction_std"], 0.0)

    def test_json_artifacts_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            write_json(path, {"undefined": float("nan")})
            self.assertIsNone(json.loads(path.read_text())["undefined"])


class SplitTests(unittest.TestCase):
    def test_patient_split_disjoint(self):
        frame = pd.DataFrame({"patient_id": [str(i) for i in range(100)], "hospital_id": [str(i % 5) for i in range(100)]})
        out = assign_patient_splits(frame, seed=3)
        assert_disjoint(out)
        self.assertEqual(len(out), len(frame))

    def test_hospital_split_disjoint(self):
        frame = pd.DataFrame({"patient_id": [str(i) for i in range(100)], "hospital_id": [str(i % 10) for i in range(100)]})
        out = assign_hospital_splits(frame, seed=3)
        self.assertTrue((out.groupby("hospital_id")["split"].nunique() == 1).all())


class CohortTests(unittest.TestCase):
    def test_mimic_landmark(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "hosp").mkdir(); (root / "icu").mkdir()
            pd.DataFrame([
                {"subject_id": 1, "gender": "F", "anchor_age": 60, "anchor_year": 2020, "anchor_year_group": "", "dod": ""},
                {"subject_id": 2, "gender": "M", "anchor_age": 17, "anchor_year": 2020, "anchor_year_group": "", "dod": ""},
            ]).to_csv(root / "hosp" / "patients.csv.gz", index=False, compression="gzip")
            pd.DataFrame([
                {"subject_id": 1, "hadm_id": 11, "hospital_expire_flag": 1},
                {"subject_id": 2, "hadm_id": 22, "hospital_expire_flag": 0},
            ]).to_csv(root / "hosp" / "admissions.csv.gz", index=False, compression="gzip")
            pd.DataFrame([
                {"subject_id": 1, "hadm_id": 11, "stay_id": 111, "first_careunit": "MICU", "last_careunit": "MICU", "intime": "2020-01-01 00:00:00", "outtime": "2020-01-02 00:00:00", "los": 1},
                {"subject_id": 2, "hadm_id": 22, "stay_id": 222, "first_careunit": "MICU", "last_careunit": "MICU", "intime": "2020-01-01 00:00:00", "outtime": "2020-01-02 00:00:00", "los": 1},
            ]).to_csv(root / "icu" / "icustays.csv.gz", index=False, compression="gzip")
            frame = build_mimic_like_cohort(root, "mimic", 6)
            self.assertEqual(len(frame), 1)
            self.assertEqual(int(frame.iloc[0].label), 1)

    def test_nwicu_uses_expired_discharge_location(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "nw_hosp").mkdir(); (root / "nw_icu").mkdir()
            pd.DataFrame([{
                "subject_id": 1, "gender": "F", "anchor_age": 60, "anchor_year": 2020,
                "anchor_year_group": "", "dod": "2020-01-02",
            }]).to_csv(root / "nw_hosp" / "patients.csv.gz", index=False, compression="gzip")
            pd.DataFrame([{
                "subject_id": 1, "hadm_id": 11, "hospital_expire_flag": 0,
                "discharge_location": "EXPIRED",
            }]).to_csv(root / "nw_hosp" / "admissions.csv.gz", index=False, compression="gzip")
            pd.DataFrame([{
                "subject_id": 1, "hadm_id": 11, "stay_id": 111, "first_careunit": "MICU",
                "last_careunit": "MICU", "intime": "2020-01-01 00:00:00",
                "outtime": "2020-01-02 00:00:00", "los": 1,
            }]).to_csv(root / "nw_icu" / "icustays.csv.gz", index=False, compression="gzip")
            frame = build_mimic_like_cohort(root, "nwicu", 6, "nw_hosp", "nw_icu")
            self.assertEqual(len(frame), 1)
            self.assertEqual(int(frame.iloc[0].label), 1)

    def test_eicu_landmark(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame([
                {"patientunitstayid": 1, "patienthealthsystemstayid": 10, "unitvisitnumber": 1, "uniquepid": "U1", "hospitaldischargeyear": 2014, "age": "65", "unitdischargeoffset": 800, "hospitaldischargestatus": "Expired", "hospitalid": 5},
                {"patientunitstayid": 3, "patienthealthsystemstayid": 30, "unitvisitnumber": 1, "uniquepid": "U1", "hospitaldischargeyear": 2015, "age": "66", "unitdischargeoffset": 900, "hospitaldischargestatus": "Alive", "hospitalid": 5},
                {"patientunitstayid": 2, "patienthealthsystemstayid": 20, "unitvisitnumber": 1, "uniquepid": "U2", "hospitaldischargeyear": 2014, "age": "16", "unitdischargeoffset": 800, "hospitaldischargestatus": "Alive", "hospitalid": 6},
            ]).to_csv(root / "patient.csv.gz", index=False, compression="gzip")
            frame = build_eicu_cohort(root, 6)
            self.assertEqual(len(frame), 1)
            self.assertEqual(int(frame.iloc[0].label), 1)


class SchemaTests(unittest.TestCase):
    def test_mapping_and_validation(self):
        self.assertEqual(canonical_concept("Heart Rate"), "heart_rate")
        self.assertEqual(canonical_concept("Non-Invasive Systolic"), "sbp")
        self.assertEqual(canonical_concept("Arterial Blood Pressure systolic"), "sbp")
        self.assertEqual(canonical_intervention("Norepinephrine infusion"), "vasopressor_norepinephrine")
        self.assertEqual(canonical_intervention("Non-invasive ventilation"), "noninvasive_ventilation")
        errors = validate_events([Event("E", 361, "x", "heart_rate", 1, 1, "", "normal")], 360)
        self.assertTrue(errors)


class SyntheticTests(unittest.TestCase):
    def test_synthetic_records(self):
        records = make_synthetic_records(50, 4)
        self.assertEqual(len(records), 50)
        self.assertTrue(all(isinstance(record, PatientRecord) and record.events for record in records))


class DerivedTaskTests(unittest.TestCase):
    def test_los_task_uses_cohort_duration_and_crops_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = [
                PatientRecord(
                    "p1",
                    "s1",
                    0,
                    "mimic",
                    split="train",
                    events=[
                        Event("E1", 10, "vital", "heart_rate", 90, 90, "bpm", "normal"),
                        Event("E2", 500, "lab", "lactate", 4.0, 4.0, "mmol/L", "high"),
                    ],
                ),
                PatientRecord("p2", "s2", 0, "mimic", split="test", events=[Event("E3", 20, "vital", "sbp", 100, 100, "", "normal")]),
            ]
            pd.DataFrame({"stay_id": ["s1", "s2"], "duration_hours": [72, 24]}).to_csv(root / "cohort.csv", index=False)
            from pces_icu.io import write_jsonl, read_jsonl

            write_jsonl(root / "events.jsonl", (record.to_dict() for record in events))
            summary = build_los_binary_task(root / "events.jsonl", root / "cohort.csv", root / "los.jsonl", threshold_hours=48)
            rows = list(read_jsonl(root / "los.jsonl"))
            self.assertEqual(summary["positives"], 1)
            self.assertEqual([row["label"] for row in rows], [1, 0])
            self.assertEqual(len(rows[0]["events"]), 1)

    def test_future_intervention_task_excludes_prevalent_and_labels_future_onset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                PatientRecord(
                    "p1",
                    "s1",
                    0,
                    "mimic",
                    split="train",
                    events=[
                        Event("E1", 10, "vital", "heart_rate", 90, 90, "", "normal"),
                        Event("E2", 700, "procedure", "invasive_ventilation", 1, 1, "", "abnormal"),
                    ],
                ),
                PatientRecord(
                    "p2",
                    "s2",
                    0,
                    "mimic",
                    split="train",
                    events=[
                        Event("E3", 20, "procedure", "invasive_ventilation", 1, 1, "", "abnormal"),
                        Event("E4", 800, "procedure", "renal_replacement", 1, 1, "", "abnormal"),
                    ],
                ),
                PatientRecord("p3", "s3", 0, "mimic", split="test", events=[Event("E5", 30, "vital", "sbp", 110, 110, "", "normal")]),
            ]
            from pces_icu.io import write_jsonl, read_jsonl

            write_jsonl(root / "events24.jsonl", (record.to_dict() for record in records))
            summary = build_future_intervention_task(root / "events24.jsonl", root / "future.jsonl")
            rows = list(read_jsonl(root / "future.jsonl"))
            self.assertEqual(summary["excluded"], 1)
            self.assertEqual([row["stay_id"] for row in rows], ["s1", "s3"])
            self.assertEqual([row["label"] for row in rows], [1, 0])
            self.assertTrue(all(event["relative_minute"] <= 360 for row in rows for event in row["events"]))


if __name__ == "__main__":
    unittest.main()
