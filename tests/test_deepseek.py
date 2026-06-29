from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pces_icu.deepseek import parse_json_response


class DeepSeekParsingTests(unittest.TestCase):
    def test_json_response(self):
        value = parse_json_response('{"probability":0.37,"evidence_event_ids":["E1"],"reason":"brief"}')
        self.assertEqual(value["probability"], 0.37)
        self.assertEqual(value["evidence_event_ids"], ["E1"])

    def test_code_fence_response(self):
        value = parse_json_response('```json\n{"probability":0.5,"evidence_event_ids":[],"reason":"x"}\n```')
        self.assertEqual(value["probability"], 0.5)

    def test_invalid_probability(self):
        with self.assertRaises(ValueError):
            parse_json_response('{"probability":1.2,"evidence_event_ids":[],"reason":"x"}')


if __name__ == "__main__":
    unittest.main()
