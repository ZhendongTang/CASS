from __future__ import annotations

import json
import math
import pickle
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .features import aggregate_record, fit_feature_names
from .io import read_jsonl, seed_everything, write_json, write_jsonl
from .metrics import binary_metrics, permutation_metrics
from .models import TransformerConfig, build_event_transformer, require_torch, symmetric_bernoulli_kl
from .schema import PatientRecord, safe_float
from .serialize import SerializationConfig, paired_views, serialize_events, stable_view_seed


def load_records(path: str | Path, split: str | None = None) -> list[PatientRecord]:
    records = [PatientRecord.from_dict(value) for value in read_jsonl(path)]
    if split is not None:
        records = [record for record in records if record.split == split]
    return records


def _prediction_rows(records: list[PatientRecord], probabilities: Iterable[float]) -> Iterable[dict[str, Any]]:
    for record, probability in zip(records, probabilities):
        yield {
            "patient_id": record.patient_id,
            "stay_id": record.stay_id,
            "dataset": record.dataset,
            "hospital_id": record.hospital_id,
            "split": record.split,
            "label": record.label,
            "probability": float(probability),
        }


class NumpyLogisticClassifier:
    """Small dependency-free baseline used for smoke tests, not the paper headline baseline."""

    def __init__(self, learning_rate: float = 0.05, epochs: int = 500, l2: float = 1e-3):
        self.learning_rate, self.epochs, self.l2 = learning_rate, epochs, l2
        self.median = self.mean = self.std = self.weight = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.median = np.nanmedian(x, axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        x = np.where(np.isnan(x), self.median, x).astype(np.float64)
        self.mean, self.std = x.mean(axis=0), x.std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        x = (x - self.mean) / self.std
        x = np.column_stack([np.ones(len(x)), x])
        self.weight = np.zeros(x.shape[1])
        pos_weight = (len(y) - y.sum()) / max(1, y.sum())
        sample_weight = np.where(y == 1, pos_weight, 1.0)
        for _ in range(self.epochs):
            logits = np.clip(x @ self.weight, -30, 30)
            probability = 1 / (1 + np.exp(-logits))
            gradient = x.T @ ((probability - y) * sample_weight) / sample_weight.sum()
            gradient[1:] += self.l2 * self.weight[1:]
            self.weight -= self.learning_rate * gradient
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.where(np.isnan(x), self.median, x).astype(np.float64)
        x = (x - self.mean) / self.std
        x = np.column_stack([np.ones(len(x)), x])
        probability = 1 / (1 + np.exp(-np.clip(x @ self.weight, -30, 30)))
        return np.column_stack([1 - probability, probability])


def train_baseline(data: str | Path, output_dir: str | Path, seed: int = 2026, model: str = "hist_gbdt") -> dict[str, Any]:
    seed_everything(seed)
    all_records = load_records(data)
    train = [record for record in all_records if record.split == "train"]
    val = [record for record in all_records if record.split == "val"]
    test = [record for record in all_records if record.split == "test"]
    if not train or not test:
        raise ValueError("train and test splits are required")
    feature_names = fit_feature_names(train)
    x_train = np.stack([aggregate_record(record, feature_names) for record in train])
    y_train = np.asarray([record.label for record in train])
    if model == "numpy_logistic":
        pipeline = NumpyLogisticClassifier().fit(x_train, y_train)
    elif model in {"hist_gbdt", "random_forest"}:
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import make_pipeline
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required for the paper baseline. Install requirements.txt.") from exc
        if model == "hist_gbdt":
            estimator = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed)
        else:
            estimator = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=seed)
        pipeline = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), estimator)
        pipeline.fit(x_train, y_train)
    else:
        raise ValueError(f"Unknown baseline model: {model}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"model": model, "seed": seed, "feature_count": len(feature_names), "splits": {}}
    for split_name, records in (("val", val), ("test", test)):
        if not records:
            continue
        x = np.stack([aggregate_record(record, feature_names) for record in records])
        probabilities = pipeline.predict_proba(x)[:, 1]
        result["splits"][split_name] = binary_metrics([record.label for record in records], probabilities)
        write_jsonl(output_dir / f"predictions_{split_name}.jsonl", _prediction_rows(records, probabilities))
    with (output_dir / "model.pkl").open("wb") as handle:
        pickle.dump({"pipeline": pipeline, "feature_names": feature_names}, handle)
    write_json(output_dir / "metrics.json", result)
    return result


def _fit_event_metadata(records: list[PatientRecord]) -> tuple[dict[str, int], dict[str, tuple[float, float]]]:
    concepts = sorted({event.concept for record in records for event in (record.events or [])})
    vocab = {"<PAD>": 0, "<UNK>": 1, **{concept: index + 2 for index, concept in enumerate(concepts)}}
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for event in record.events or []:
            value = safe_float(event.canonical_value)
            if value is not None:
                values[event.concept].append(value)
    stats = {}
    for concept, items in values.items():
        array = np.asarray(items, dtype=float)
        std = float(array.std())
        stats[concept] = (float(array.mean()), std if std > 1e-6 else 1.0)
    return vocab, stats


def _event_tensors(record: PatientRecord, vocab: dict[str, int], stats: dict[str, tuple[float, float]], max_length: int):
    selected = sorted(record.events or [], key=lambda event: (event.relative_minute, event.event_id))[-max_length:]
    concept, numeric = [], []
    previous = 0
    flag_code = {"low": -1.0, "normal": 0.0, "unknown": 0.0, "abnormal": 1.0, "high": 1.0}
    for event in selected:
        concept.append(vocab.get(event.concept, 1))
        raw = safe_float(event.canonical_value)
        mean, std = stats.get(event.concept, (0.0, 1.0))
        z = 0.0 if raw is None else float(np.clip((raw - mean) / std, -10, 10))
        delta = math.log1p(max(0, event.relative_minute - previous)) / math.log1p(360)
        numeric.append([z, delta, flag_code.get(event.abnormal_flag, 0.0)])
        previous = event.relative_minute
    if not concept:
        concept, numeric = [1], [[0.0, 0.0, 0.0]]
    return concept, numeric


def train_event_transformer(data: str | Path, output_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    torch = require_torch()
    seed = int(config.get("seed", 2026))
    seed_everything(seed)
    records = load_records(data)
    train = [record for record in records if record.split == "train"]
    val = [record for record in records if record.split == "val"]
    test = [record for record in records if record.split == "test"]
    vocab, stats = _fit_event_metadata(train)
    cfg = TransformerConfig(
        vocab_size=len(vocab), hidden_size=int(config.get("hidden_size", 192)), layers=int(config.get("layers", 4)),
        heads=int(config.get("heads", 6)), dropout=float(config.get("dropout", 0.1)), max_length=int(config.get("max_length", 512)),
    )
    model = build_event_transformer(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch_size = int(config.get("batch_size", 32))

    def batches(items, shuffle=False):
        indices = np.arange(len(items))
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            chosen = [items[index] for index in indices[start : start + batch_size]]
            seqs = [_event_tensors(record, vocab, stats, cfg.max_length) for record in chosen]
            length = max(len(value[0]) for value in seqs)
            ids = torch.zeros((len(chosen), length), dtype=torch.long)
            nums = torch.zeros((len(chosen), length, 3), dtype=torch.float32)
            mask = torch.zeros((len(chosen), length), dtype=torch.bool)
            for row, (concept, numeric) in enumerate(seqs):
                ids[row, : len(concept)] = torch.tensor(concept)
                nums[row, : len(concept)] = torch.tensor(numeric)
                mask[row, : len(concept)] = True
            labels = torch.tensor([record.label for record in chosen], dtype=torch.float32)
            yield chosen, ids.to(device), nums.to(device), mask.to(device), labels.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 3e-4)), weight_decay=float(config.get("weight_decay", 0.01)))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(train) - sum(r.label for r in train)) / max(1, sum(r.label for r in train))], device=device))
    best_state, best_auprc = None, -1.0
    epochs = int(config.get("epochs", 20))
    patience, stale = int(config.get("patience", 4)), 0

    def predict(items):
        model.eval(); probs = []
        with torch.no_grad():
            for _, ids, nums, mask, _ in batches(items):
                probs.extend(torch.sigmoid(model(ids, nums, mask)).cpu().tolist())
        return np.asarray(probs)

    for _ in range(epochs):
        model.train()
        for _, ids, nums, mask, labels in batches(train, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(ids, nums, mask), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_prob = predict(val)
        score = binary_metrics([record.label for record in val], val_prob)["auprc"] if val else 0.0
        if score > best_auprc:
            best_auprc, stale = score, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    result = {"model": "event_transformer", "seed": seed, "config": asdict(cfg), "splits": {}}
    for split_name, items in (("val", val), ("test", test)):
        if items:
            probabilities = predict(items)
            result["splits"][split_name] = binary_metrics([record.label for record in items], probabilities)
            write_jsonl(output_dir / f"predictions_{split_name}.jsonl", _prediction_rows(items, probabilities))
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "vocab": vocab, "stats": stats}, output_dir / "model.pt")
    write_json(output_dir / "metrics.json", result)
    return result


def _build_hf_model(config: dict[str, Any]):
    torch = require_torch()
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("transformers is required for LLM training") from exc
    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=bool(config.get("trust_remote_code", False)))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {"num_labels": 1, "trust_remote_code": bool(config.get("trust_remote_code", False))}
    if bool(config.get("load_in_4bit", True)) and torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        kwargs["device_map"] = "auto"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
    if bool(config.get("use_lora", True)):
        try:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise RuntimeError("peft is required when use_lora=true") from exc
        if bool(config.get("load_in_4bit", True)):
            model = prepare_model_for_kbit_training(model)
        lora = LoraConfig(
            r=int(config.get("lora_r", 16)), lora_alpha=int(config.get("lora_alpha", 32)),
            lora_dropout=float(config.get("lora_dropout", 0.05)), bias="none", task_type=TaskType.SEQ_CLS,
            target_modules=config.get("lora_targets", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        )
        model = get_peft_model(model, lora)
    return tokenizer, model


def train_llm(data: str | Path, output_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    torch = require_torch()
    seed = int(config.get("seed", 2026)); seed_everything(seed)
    records = load_records(data)
    train = [record for record in records if record.split == "train"]
    val = [record for record in records if record.split == "val"]
    test = [record for record in records if record.split == "test"]
    tokenizer, model = _build_hf_model(config)
    device = next(model.parameters()).device
    if device.type == "cpu" and torch.cuda.is_available():
        model.to("cuda"); device = next(model.parameters()).device
    ser_cfg = SerializationConfig(
        include_delta_t=not bool(config.get("no_delta_t", False)),
        include_set_markers=not bool(config.get("flat_serialization", False)),
        permute_within_sets=not bool(config.get("no_permutation", False)),
        shuffle_across_timestamps=bool(config.get("shuffle_across_timestamps", False)),
        max_events=int(config.get("max_events", 512)), max_chars=int(config.get("max_chars", 24_000)),
    )
    max_length, batch_size = int(config.get("max_length", 2048)), int(config.get("batch_size", 2))
    lambda_consistency = float(config.get("lambda_consistency", 0.5))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(config.get("learning_rate", 2e-4)), weight_decay=float(config.get("weight_decay", 0.01)))
    positives = sum(record.label for record in train)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(train) - positives) / max(1, positives)], device=device))

    def tokenized(texts):
        return {key: value.to(device) for key, value in tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").items()}

    def train_batches(epoch):
        indices = np.arange(len(train)); np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            chosen = [train[index] for index in indices[start : start + batch_size]]
            views = [paired_views(record.events or [], ser_cfg, stable_view_seed(record.patient_id, seed + epoch)) for record in chosen]
            yield chosen, tokenized([view[0] for view in views]), tokenized([view[1] for view in views])

    def predict(items, permutation_seed=0):
        model.eval(); probabilities = []
        fixed = SerializationConfig(**{**asdict(ser_cfg), "permute_within_sets": permutation_seed != 0})
        with torch.no_grad():
            for start in range(0, len(items), batch_size):
                chosen = items[start : start + batch_size]
                texts = [serialize_events(record.events or [], fixed, stable_view_seed(record.patient_id, permutation_seed)) for record in chosen]
                logits = model(**tokenized(texts)).logits.squeeze(-1)
                probabilities.extend(torch.sigmoid(logits).float().cpu().tolist())
        return np.asarray(probabilities)

    epochs = int(config.get("epochs", 3))
    accumulation = int(config.get("gradient_accumulation_steps", 8))
    model.train(); optimizer.zero_grad(set_to_none=True)
    global_step = 0
    for epoch in range(epochs):
        for chosen, view_a, view_b in train_batches(epoch):
            labels = torch.tensor([record.label for record in chosen], dtype=torch.float32, device=device)
            logits_a = model(**view_a).logits.squeeze(-1)
            logits_b = model(**view_b).logits.squeeze(-1)
            loss = (loss_fn(logits_a, labels) + loss_fn(logits_b, labels) + lambda_consistency * symmetric_bernoulli_kl(logits_a, logits_b)) / accumulation
            loss.backward(); global_step += 1
            if global_step % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
    if global_step % accumulation:
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir / "model"); tokenizer.save_pretrained(output_dir / "model")
    result: dict[str, Any] = {"model": config["model_name"], "seed": seed, "variant": config.get("variant", "pces"), "serialization": asdict(ser_cfg), "splits": {}}
    for split_name, items in (("val", val), ("test", test)):
        if items:
            probabilities = predict(items)
            result["splits"][split_name] = binary_metrics([record.label for record in items], probabilities)
            write_jsonl(output_dir / f"predictions_{split_name}.jsonl", _prediction_rows(items, probabilities))
    if test:
        matrix = np.stack([predict(test, permutation_seed=seed + index + 1) for index in range(int(config.get("permutation_evaluations", 10)))], axis=1)
        result["permutation"] = permutation_metrics(matrix)
        np.save(output_dir / "permutation_probabilities.npy", matrix)
    write_json(output_dir / "metrics.json", result)
    return result


def _hospital_metrics(records: list[PatientRecord], probabilities: np.ndarray) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record.hospital_id:
            grouped[record.hospital_id].append(index)
    per_hospital: dict[str, Any] = {}
    for hospital, indices in grouped.items():
        y = [records[index].label for index in indices]
        p = [probabilities[index] for index in indices]
        per_hospital[hospital] = binary_metrics(y, p)
    valid_auroc = [value["auroc"] for value in per_hospital.values() if math.isfinite(value["auroc"])]
    valid_auprc = [value["auprc"] for value in per_hospital.values() if math.isfinite(value["auprc"])]
    return {
        "hospital_count": len(per_hospital),
        "macro_auroc": float(np.mean(valid_auroc)) if valid_auroc else math.nan,
        "macro_auprc": float(np.mean(valid_auprc)) if valid_auprc else math.nan,
        "per_hospital": per_hospital,
    }


def evaluate_baseline_checkpoint(data: str | Path, model_dir: str | Path, output_dir: str | Path, split: str = "test") -> dict[str, Any]:
    records = load_records(data, split)
    with (Path(model_dir) / "model.pkl").open("rb") as handle:
        artifact = pickle.load(handle)
    x = np.stack([aggregate_record(record, artifact["feature_names"]) for record in records])
    probabilities = artifact["pipeline"].predict_proba(x)[:, 1]
    result = {"split": split, "metrics": binary_metrics([record.label for record in records], probabilities), "hospitals": _hospital_metrics(records, probabilities)}
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "predictions.jsonl", _prediction_rows(records, probabilities)); write_json(output_dir / "metrics.json", result)
    return result


def evaluate_transformer_checkpoint(data: str | Path, model_dir: str | Path, output_dir: str | Path, split: str = "test", batch_size: int = 64) -> dict[str, Any]:
    torch = require_torch()
    artifact = torch.load(Path(model_dir) / "model.pt", map_location="cpu", weights_only=False)
    config = TransformerConfig(**artifact["config"])
    model = build_event_transformer(config); model.load_state_dict(artifact["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device); model.eval()
    records = load_records(data, split); probabilities = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            chosen = records[start : start + batch_size]
            seqs = [_event_tensors(record, artifact["vocab"], artifact["stats"], config.max_length) for record in chosen]
            length = max(len(value[0]) for value in seqs)
            ids = torch.zeros((len(chosen), length), dtype=torch.long); nums = torch.zeros((len(chosen), length, 3)); mask = torch.zeros((len(chosen), length), dtype=torch.bool)
            for row, (concept, numeric) in enumerate(seqs):
                ids[row, : len(concept)] = torch.tensor(concept); nums[row, : len(concept)] = torch.tensor(numeric); mask[row, : len(concept)] = True
            logits = model(ids.to(device), nums.to(device), mask.to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
    probabilities = np.asarray(probabilities)
    result = {"split": split, "metrics": binary_metrics([record.label for record in records], probabilities), "hospitals": _hospital_metrics(records, probabilities)}
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "predictions.jsonl", _prediction_rows(records, probabilities)); write_json(output_dir / "metrics.json", result)
    return result


def _load_saved_llm(model_dir: Path, config: dict[str, Any]):
    torch = require_torch()
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for LLM evaluation") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=bool(config.get("trust_remote_code", False)))
    kwargs: dict[str, Any] = {"trust_remote_code": bool(config.get("trust_remote_code", False))}
    if torch.cuda.is_available():
        kwargs.update({"torch_dtype": torch.bfloat16, "device_map": "auto"})
    if (model_dir / "adapter_config.json").exists():
        try:
            from peft import PeftConfig, PeftModel
        except ImportError as exc:
            raise RuntimeError("peft is required to load a LoRA checkpoint") from exc
        peft_cfg = PeftConfig.from_pretrained(model_dir)
        base = AutoModelForSequenceClassification.from_pretrained(peft_cfg.base_model_name_or_path, num_labels=1, **kwargs)
        model = PeftModel.from_pretrained(base, model_dir)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def evaluate_llm_checkpoint(data: str | Path, model_dir: str | Path, output_dir: str | Path, config: dict[str, Any], split: str = "test") -> dict[str, Any]:
    torch = require_torch(); records = load_records(data, split)
    tokenizer, model = _load_saved_llm(Path(model_dir), config)
    device = next(model.parameters()).device; model.eval()
    ser_cfg = SerializationConfig(
        include_delta_t=not bool(config.get("no_delta_t", False)), include_set_markers=not bool(config.get("flat_serialization", False)),
        permute_within_sets=False, shuffle_across_timestamps=bool(config.get("shuffle_across_timestamps", False)),
        max_events=int(config.get("max_events", 512)), max_chars=int(config.get("max_chars", 24_000)),
    )
    batch_size, max_length = int(config.get("eval_batch_size", config.get("batch_size", 2))), int(config.get("max_length", 2048))

    def predict(permutation_seed=0):
        cfg = SerializationConfig(**{**asdict(ser_cfg), "permute_within_sets": permutation_seed != 0})
        values = []
        with torch.no_grad():
            for start in range(0, len(records), batch_size):
                chosen = records[start : start + batch_size]
                texts = [serialize_events(record.events or [], cfg, stable_view_seed(record.patient_id, permutation_seed)) for record in chosen]
                batch = {k: v.to(device) for k, v in tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").items()}
                values.extend(torch.sigmoid(model(**batch).logits.squeeze(-1)).float().cpu().tolist())
        return np.asarray(values)

    probabilities = predict()
    result: dict[str, Any] = {"split": split, "metrics": binary_metrics([record.label for record in records], probabilities), "hospitals": _hospital_metrics(records, probabilities)}
    permutations = int(config.get("permutation_evaluations", 10))
    if permutations > 1:
        matrix = np.stack([predict(int(config.get("seed", 2026)) + index + 1) for index in range(permutations)], axis=1)
        result["permutation"] = permutation_metrics(matrix)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "predictions.jsonl", _prediction_rows(records, probabilities)); write_json(output_dir / "metrics.json", result)
    return result
