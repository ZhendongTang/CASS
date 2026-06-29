from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pces_icu.multicenter_llm import (  # noqa: E402
    _load_base,
    _inference_variant,
    _model_layout,
    _records,
    _text,
    _uses_two_views,
    parser,
)
from pces_icu.schema import Event, PatientRecord  # noqa: E402
from pces_icu.serialize import extract_event_ids  # noqa: E402


class _DummyModule:
    def __init__(self, names):
        self._names = names

    def named_modules(self):
        return iter(self._names)


class MulticenterVariantTests(unittest.TestCase):
    def test_variant_semantics(self):
        self.assertFalse(_uses_two_views("xml"))
        self.assertTrue(_uses_two_views("xml_aug"))
        self.assertTrue(_uses_two_views("xml_kl"))
        self.assertEqual(_inference_variant("xml_aug"), "xml")
        self.assertEqual(_inference_variant("xml_kl"), "xml")
        self.assertEqual(_inference_variant("summary"), "summary")
        self.assertEqual(_inference_variant("flat"), "flat")

    def test_sampling_seed_is_decoupled_from_model_seed(self):
        args = parser().parse_args([
            "train", "--data", "data.jsonl.gz", "--model", "model",
            "--output", "out", "--variant", "flat", "--seed", "44",
        ])
        self.assertEqual(args.seed, 44)
        self.assertEqual(args.sample_seed, 2026)

    def test_training_parser_accepts_summary_and_reweighting(self):
        args = parser().parse_args([
            "train", "--data", "data.jsonl.gz", "--model", "model",
            "--output", "out", "--variant", "summary", "--auto-pos-weight",
            "--pos-weight-cap", "4", "--balance-train",
            "--train-eval-records", "128", "--task-prompt", "Task: predict LOS.\n",
        ])
        self.assertEqual(args.variant, "summary")
        self.assertTrue(args.auto_pos_weight)
        self.assertEqual(args.pos_weight_cap, 4)
        self.assertTrue(args.balance_train)
        self.assertEqual(args.train_eval_records, 128)
        self.assertEqual(args.task_prompt, "Task: predict LOS.\n")

    @patch("pces_icu.multicenter_llm.read_jsonl")
    def test_balanced_records_sampling(self, read_jsonl):
        read_jsonl.return_value = [
            {"dataset": "x", "patient_id": f"n{index}", "stay_id": f"sn{index}", "label": 0, "split": "train", "events": []}
            for index in range(10)
        ] + [
            {"dataset": "x", "patient_id": f"p{index}", "stay_id": f"sp{index}", "label": 1, "split": "train", "events": []}
            for index in range(2)
        ]
        records = _records("dummy.jsonl.gz", "train", 8, 7, balance=True)
        self.assertEqual(len(records), 8)
        self.assertEqual(sum(record.label for record in records), 4)

    def test_glm_layout(self):
        layout = _model_layout(_DummyModule([
            ("", object()),
            ("classifier_head", object()),
            ("transformer.layer.query_key_value", object()),
        ]))
        self.assertEqual(layout["head_names"], ["classifier_head"])
        self.assertEqual(layout["lora_targets"], ["query_key_value"])

    def test_llama_gemma_layout(self):
        layout = _model_layout(_DummyModule([
            ("", object()),
            ("score", object()),
            ("model.layers.0.self_attn.q_proj", object()),
            ("model.layers.0.self_attn.k_proj", object()),
            ("model.layers.0.self_attn.v_proj", object()),
            ("model.layers.0.self_attn.o_proj", object()),
        ]))
        self.assertEqual(layout["head_names"], ["score"])
        self.assertEqual(layout["lora_targets"], ["q_proj", "k_proj", "v_proj", "o_proj"])

    def test_forced_permutation_preserves_evidence(self):
        record = PatientRecord(
            dataset="mimic",
            patient_id="P1",
            stay_id="S1",
            label=0,
            split="test",
            events=[
                Event("E1", 10, "vital", "heart_rate", 120, 120.0, "bpm", "high"),
                Event("E2", 10, "vital", "sbp", 85, 85.0, "mmHg", "low"),
                Event("E3", 10, "vital", "spo2", 90, 90.0, "%", "low"),
            ],
        )
        canonical = _text(record, "xml", 1)
        permuted = _text(record, "xml", 2, force_permutation=True)
        self.assertEqual(sorted(extract_event_ids(canonical)), sorted(extract_event_ids(permuted)))

    def test_summary_format_uses_task_prompt(self):
        record = PatientRecord(
            dataset="mimic",
            patient_id="P1",
            stay_id="S1",
            label=1,
            split="train",
            events=[
                Event("E1", 10, "vital", "heart_rate", 90, 90.0, "bpm", "normal"),
                Event("E2", 20, "vital", "heart_rate", 120, 120.0, "bpm", "high"),
            ],
        )
        text = _text(record, "summary", 1, task_prompt="Task: predict LOS.\n")
        self.assertTrue(text.startswith("Task: predict LOS.\nClinTime-FeatureSummary/1"))
        self.assertIn("GLOBAL|events=2", text)
        self.assertIn("HR|n=2", text)
        self.assertIn("mean=105", text)

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_tokenizer_falls_back_to_slow_implementation(self, from_pretrained):
        slow = MagicMock()
        slow.pad_token_id = 0
        from_pretrained.side_effect = [ValueError("incompatible tokenizer.json"), slow]
        with patch("transformers.AutoModelForSequenceClassification.from_pretrained") as model_loader:
            model = MagicMock()
            model.named_modules.return_value = iter([
                ("", model),
                ("score", MagicMock()),
                ("model.layers.0.self_attn.q_proj", MagicMock()),
                ("model.layers.0.self_attn.k_proj", MagicMock()),
                ("model.layers.0.self_attn.v_proj", MagicMock()),
                ("model.layers.0.self_attn.o_proj", MagicMock()),
            ])
            model.config = MagicMock()
            model_loader.return_value = model
            with patch("pces_icu.multicenter_llm._initialize_classification_heads"):
                tokenizer, _, _ = _load_base("dummy", training=False)
        self.assertIs(tokenizer, slow)
        self.assertEqual(from_pretrained.call_args_list[0].kwargs["use_fast"], True)
        self.assertEqual(from_pretrained.call_args_list[1].kwargs["use_fast"], False)


if __name__ == "__main__":
    unittest.main()
