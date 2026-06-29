from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import read_jsonl, write_json, write_jsonl
from .metrics import binary_metrics, permutation_metrics
from .schema import PatientRecord
from .serialize import (
    ClinTimeFeatureSummaryConfig,
    ClinTimeXMLConfig,
    SerializationConfig,
    extract_event_ids,
    serialize_clintime_feature_summary,
    serialize_clintime_xml,
    serialize_events,
    stable_view_seed,
)


SYSTEM_PROMPT = """You are evaluating retrospective, de-identified ICU records for a research benchmark.
Estimate the probability of in-hospital mortality using only the events shown from the first six ICU hours.
Do not infer events that are not present. Return one JSON object with exactly these fields:
probability (number from 0 to 1), evidence_event_ids (array of up to 5 IDs present in the input), reason (brief string).
This is research model output, not clinical advice."""


class DeepSeekClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str = "deepseek-chat", timeout: int = 120):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(self, event_text: str, retries: int = 5) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Return JSON only.\n\nPATIENT_EVENT_STREAM:\n" + event_text},
            ],
            "temperature": 0,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        endpoint = self.base_url + "/chat/completions"
        for attempt in range(retries):
            request = urllib.request.Request(
                endpoint,
                data=data,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                parsed = parse_json_response(content)
                parsed["usage"] = result.get("usage", {})
                parsed["request_id"] = result.get("id", "")
                return parsed
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt + 1 == retries:
                    raise RuntimeError(f"DeepSeek HTTP {exc.code}: {body[:500]}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
                if attempt + 1 == retries:
                    raise RuntimeError(f"DeepSeek request failed: {exc}") from exc
            time.sleep(min(30, 2**attempt + random.random()))
        raise AssertionError("unreachable")

    def complete_summary(self, summary_text: str, task_prompt: str, retries: int = 5) -> dict[str, Any]:
        system_prompt = "Output only compact JSON with fields probability and reason. Keep reason under 20 words."
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{task_prompt.rstrip()}\nReturn JSON only.\n{summary_text}"},
            ],
            "temperature": 0,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        reasoning_effort = os.environ.get("DEEPSEEK_REASONING_EFFORT")
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if os.environ.get("DEEPSEEK_THINKING", "").lower() in {"1", "true", "enabled"}:
            payload["thinking"] = {"type": "enabled"}
        data = json.dumps(payload).encode("utf-8")
        endpoint = self.base_url + "/chat/completions"
        for attempt in range(retries):
            request = urllib.request.Request(
                endpoint,
                data=data,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                parsed = parse_summary_json_response(content)
                parsed["usage"] = result.get("usage", {})
                parsed["request_id"] = result.get("id", "")
                return parsed
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt + 1 == retries:
                    raise RuntimeError(f"DeepSeek HTTP {exc.code}: {body[:500]}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
                if attempt + 1 == retries:
                    raise RuntimeError(f"DeepSeek request failed: {exc}") from exc
            time.sleep(min(30, 2**attempt + random.random()))
        raise AssertionError("unreachable")


def parse_json_response(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I | re.S)
    value = json.loads(cleaned)
    probability = float(value["probability"])
    if not 0 <= probability <= 1:
        raise ValueError(f"probability outside [0,1]: {probability}")
    evidence = value.get("evidence_event_ids", [])
    if not isinstance(evidence, list):
        raise ValueError("evidence_event_ids must be an array")
    return {"probability": probability, "evidence_event_ids": [str(item) for item in evidence[:5]], "reason": str(value.get("reason", ""))[:1000]}


def parse_summary_json_response(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I | re.S)
    value = json.loads(cleaned)
    probability = float(value["probability"])
    if not 0 <= probability <= 1:
        raise ValueError(f"probability outside [0,1]: {probability}")
    return {"probability": probability, "reason": str(value.get("reason", ""))[:1000]}


def _sample(records: list[PatientRecord], maximum: int, seed: int) -> list[PatientRecord]:
    ordered = sorted(records, key=lambda record: hashlib.blake2b(f"{seed}:{record.patient_id}".encode(), digest_size=8).digest())
    return ordered[:maximum] if maximum > 0 else ordered


def run_deepseek_pilot(
    data: str | Path,
    output_dir: str | Path,
    split: str = "test",
    max_records: int = 100,
    permutations: int = 3,
    seed: int = 2026,
    workers: int = 4,
    model: str = "deepseek-chat",
    max_events: int = 128,
    max_chars: int = 12_000,
) -> dict[str, Any]:
    records = [PatientRecord.from_dict(value) for value in read_jsonl(data)]
    records = _sample([record for record in records if record.split == split], max_records, seed)
    if not records:
        raise ValueError(f"No records found for split={split}")
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "responses.jsonl"
    cached: dict[str, dict[str, Any]] = {}
    if raw_path.exists():
        cached = {row["task_id"]: row for row in read_jsonl(raw_path) if row.get("status") == "ok"}

    flat_cfg = SerializationConfig(include_delta_t=True, include_set_markers=False, permute_within_sets=False, max_events=max_events, max_chars=max_chars)
    pces_cfg = SerializationConfig(include_delta_t=True, include_set_markers=True, permute_within_sets=True, max_events=max_events, max_chars=max_chars)
    tasks: list[tuple[str, PatientRecord, str, int, str]] = []
    for record in records:
        flat_id = f"{record.patient_id}|flat|0"
        if flat_id not in cached:
            tasks.append((flat_id, record, "flat", 0, serialize_events(record.events or [], flat_cfg, stable_view_seed(record.patient_id, seed))))
        for index in range(permutations):
            task_id = f"{record.patient_id}|pces|{index}"
            if task_id not in cached:
                text = serialize_events(record.events or [], pces_cfg, stable_view_seed(record.patient_id, seed + index + 1))
                tasks.append((task_id, record, "pces", index, text))

    client = DeepSeekClient(model=model)

    def execute(task):
        task_id, record, variant, permutation, text = task
        try:
            response = client.complete(text)
            valid_ids = set(extract_event_ids(text))
            cited = response["evidence_event_ids"]
            return {
                "task_id": task_id, "status": "ok", "patient_id": record.patient_id, "stay_id": record.stay_id,
                "dataset": record.dataset, "hospital_id": record.hospital_id, "split": record.split, "label": record.label,
                "variant": variant, "permutation": permutation, "probability": response["probability"],
                "evidence_event_ids": cited, "evidence_valid": all(item in valid_ids for item in cited),
                "reason": response["reason"], "usage": response.get("usage", {}), "request_id": response.get("request_id", ""),
            }
        except Exception as exc:
            return {"task_id": task_id, "status": "error", "patient_id": record.patient_id, "label": record.label, "variant": variant, "permutation": permutation, "error": str(exc)[:1000]}

    new_rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for row in executor.map(execute, tasks):
            new_rows.append(row)
            print(json.dumps({key: row.get(key) for key in ("task_id", "status", "probability", "error") if row.get(key) is not None}, ensure_ascii=False), flush=True)
    combined = list(cached.values()) + new_rows
    write_jsonl(raw_path, sorted(combined, key=lambda row: row["task_id"]))
    ok = [row for row in combined if row.get("status") == "ok"]
    by_patient: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in ok:
        by_patient[row["patient_id"]].setdefault(row["variant"], []).append(row)
    complete = [record for record in records if len(by_patient[record.patient_id].get("flat", [])) == 1 and len(by_patient[record.patient_id].get("pces", [])) == permutations]
    labels = [record.label for record in complete]
    flat_probability = [by_patient[record.patient_id]["flat"][0]["probability"] for record in complete]
    pces_matrix = np.asarray([[row["probability"] for row in sorted(by_patient[record.patient_id]["pces"], key=lambda x: x["permutation"])] for record in complete]) if complete else np.empty((0, permutations))
    prompt_tokens = sum(int(row.get("usage", {}).get("prompt_tokens", 0) or 0) for row in ok)
    completion_tokens = sum(int(row.get("usage", {}).get("completion_tokens", 0) or 0) for row in ok)
    result: dict[str, Any] = {
        "model": model, "split": split, "requested_records": len(records), "complete_records": len(complete),
        "calls_total": len(combined), "calls_ok": len(ok), "calls_error": len(combined) - len(ok),
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "evidence_valid_rate": float(np.mean([row["evidence_valid"] for row in ok])) if ok else None,
    }
    if complete:
        result["flat"] = binary_metrics(labels, flat_probability)
        result["pces_ensemble"] = binary_metrics(labels, pces_matrix.mean(axis=1))
        if permutations > 1:
            result["pces_permutation"] = permutation_metrics(pces_matrix)
    write_json(output_dir / "metrics.json", result)
    return result


def run_deepseek_txml_pilot(
    data: str | Path,
    output_dir: str | Path,
    split: str = "test",
    max_records: int = 100,
    permutations: int = 3,
    seed: int = 2026,
    workers: int = 4,
    model: str = "deepseek-chat",
    max_nodes: int = 128,
    max_chars: int = 12_000,
    bucket_minutes: int = 60,
    canonical_order: bool = False,
) -> dict[str, Any]:
    records = [PatientRecord.from_dict(value) for value in read_jsonl(data)]
    records = _sample([record for record in records if record.split == split], max_records, seed)
    if not records:
        raise ValueError(f"No records found for split={split}")
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "responses.jsonl"
    cached: dict[str, dict[str, Any]] = {}
    if raw_path.exists():
        cached = {row["task_id"]: row for row in read_jsonl(raw_path) if row.get("status") == "ok"}
    config = ClinTimeXMLConfig(
        bucket_minutes=bucket_minutes,
        permute_within_blocks=not canonical_order,
        max_nodes=max_nodes,
        max_chars=max_chars,
    )
    variant = "txml_canonical" if canonical_order else "txml"
    if canonical_order and permutations != 1:
        raise ValueError("canonical_order requires permutations=1 because every view is identical by construction")
    tasks: list[tuple[str, PatientRecord, int, str]] = []
    for record in records:
        for index in range(permutations):
            task_id = f"{record.patient_id}|{variant}|{index}"
            if task_id not in cached:
                text = serialize_clintime_xml(
                    record.events or [], config, stable_view_seed(record.patient_id, seed + index + 1)
                )
                tasks.append((task_id, record, index, text))
    client = DeepSeekClient(model=model)

    def execute(task):
        task_id, record, permutation, text = task
        try:
            response = client.complete(text)
            valid_ids = set(extract_event_ids(text))
            cited = response["evidence_event_ids"]
            return {
                "task_id": task_id, "status": "ok", "patient_id": record.patient_id, "stay_id": record.stay_id,
                "dataset": record.dataset, "hospital_id": record.hospital_id, "split": record.split,
                "label": record.label, "variant": variant, "permutation": permutation,
                "probability": response["probability"], "evidence_event_ids": cited,
                "evidence_valid": all(item in valid_ids for item in cited), "reason": response["reason"],
                "input_chars": len(text), "input_event_ids": len(valid_ids),
                "usage": response.get("usage", {}), "request_id": response.get("request_id", ""),
            }
        except Exception as exc:
            return {
                "task_id": task_id, "status": "error", "patient_id": record.patient_id,
                "label": record.label, "variant": variant, "permutation": permutation,
                "input_chars": len(text), "error": str(exc)[:1000],
            }

    new_rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for row in executor.map(execute, tasks):
            new_rows.append(row)
            print(json.dumps({key: row.get(key) for key in ("task_id", "status", "probability", "error") if row.get(key) is not None}, ensure_ascii=False), flush=True)
    combined_by_id = dict(cached)
    combined_by_id.update({row["task_id"]: row for row in new_rows})
    combined = sorted(combined_by_id.values(), key=lambda row: row["task_id"])
    write_jsonl(raw_path, combined)
    ok = [row for row in combined if row.get("status") == "ok"]
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ok:
        by_patient[row["patient_id"]].append(row)
    complete = [record for record in records if len(by_patient[record.patient_id]) == permutations]
    labels = [record.label for record in complete]
    matrix = np.asarray(
        [[row["probability"] for row in sorted(by_patient[record.patient_id], key=lambda value: value["permutation"])] for record in complete]
    ) if complete else np.empty((0, permutations))
    result: dict[str, Any] = {
        "model": model,
        "serialization": "ClinTime-XML/1.1-canonical" if canonical_order else "ClinTime-XML/1.0",
        "canonical_order": canonical_order, "split": split,
        "requested_records": len(records), "complete_records": len(complete),
        "calls_total": len(combined), "calls_ok": len(ok), "calls_error": len(combined) - len(ok),
        "prompt_tokens": sum(int(row.get("usage", {}).get("prompt_tokens", 0) or 0) for row in ok),
        "completion_tokens": sum(int(row.get("usage", {}).get("completion_tokens", 0) or 0) for row in ok),
        "mean_input_chars": float(np.mean([row.get("input_chars", 0) for row in ok])) if ok else None,
        "mean_input_event_ids": float(np.mean([row.get("input_event_ids", 0) for row in ok])) if ok else None,
        "evidence_valid_rate": float(np.mean([row["evidence_valid"] for row in ok])) if ok else None,
    }
    if complete:
        result[variant] = binary_metrics(labels, matrix.mean(axis=1))
        if permutations > 1:
            result["txml_permutation"] = permutation_metrics(matrix)
    write_json(output_dir / "metrics.json", result)
    return result


def run_deepseek_summary_pilot(
    data: str | Path,
    output_dir: str | Path,
    task_prompt: str,
    split: str = "test",
    max_records: int = 100,
    seed: int = 2026,
    workers: int = 4,
    model: str = "deepseek-chat",
    max_rows: int = 80,
    max_chars: int = 3_000,
) -> dict[str, Any]:
    records = [PatientRecord.from_dict(value) for value in read_jsonl(data)]
    records = _sample([record for record in records if record.split == split], max_records, seed)
    if not records:
        raise ValueError(f"No records found for split={split}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "responses.jsonl"
    cached: dict[str, dict[str, Any]] = {}
    if raw_path.exists():
        cached = {row["task_id"]: row for row in read_jsonl(raw_path) if row.get("status") == "ok"}

    config = ClinTimeFeatureSummaryConfig(max_rows=max_rows, max_chars=max_chars)
    tasks: list[tuple[str, PatientRecord, str]] = []
    for record in records:
        task_id = f"{record.patient_id}|summary|0"
        if task_id not in cached:
            text = serialize_clintime_feature_summary(record.events or [], config)
            tasks.append((task_id, record, text))
    client = DeepSeekClient(model=model)

    def execute(task):
        task_id, record, text = task
        try:
            response = client.complete_summary(text, task_prompt)
            return {
                "task_id": task_id,
                "status": "ok",
                "patient_id": record.patient_id,
                "stay_id": record.stay_id,
                "dataset": record.dataset,
                "hospital_id": record.hospital_id,
                "split": record.split,
                "label": record.label,
                "variant": "summary",
                "probability": response["probability"],
                "reason": response["reason"],
                "input_chars": len(text),
                "usage": response.get("usage", {}),
                "request_id": response.get("request_id", ""),
            }
        except Exception as exc:
            return {
                "task_id": task_id,
                "status": "error",
                "patient_id": record.patient_id,
                "label": record.label,
                "variant": "summary",
                "input_chars": len(text),
                "error": str(exc)[:1000],
            }

    new_rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for row in executor.map(execute, tasks):
            new_rows.append(row)
            print(
                json.dumps(
                    {key: row.get(key) for key in ("task_id", "status", "probability", "error") if row.get(key) is not None},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    combined_by_id = dict(cached)
    combined_by_id.update({row["task_id"]: row for row in new_rows})
    combined = sorted(combined_by_id.values(), key=lambda row: row["task_id"])
    write_jsonl(raw_path, combined)
    ok = [row for row in combined if row.get("status") == "ok"]
    complete_by_patient = {row["patient_id"]: row for row in ok}
    complete = [record for record in records if record.patient_id in complete_by_patient]
    labels = [record.label for record in complete]
    probabilities = [complete_by_patient[record.patient_id]["probability"] for record in complete]
    result: dict[str, Any] = {
        "model": model,
        "serialization": "ClinTime-FeatureSummary/1",
        "task_prompt": task_prompt,
        "split": split,
        "requested_records": len(records),
        "complete_records": len(complete),
        "calls_total": len(combined),
        "calls_ok": len(ok),
        "calls_error": len(combined) - len(ok),
        "prompt_tokens": sum(int(row.get("usage", {}).get("prompt_tokens", 0) or 0) for row in ok),
        "completion_tokens": sum(int(row.get("usage", {}).get("completion_tokens", 0) or 0) for row in ok),
        "mean_input_chars": float(np.mean([row.get("input_chars", 0) for row in ok])) if ok else None,
    }
    if complete:
        result["summary"] = binary_metrics(labels, probabilities)
    write_json(output_dir / "metrics.json", result)
    return result
