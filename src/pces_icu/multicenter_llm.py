from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from .io import read_jsonl, seed_everything, write_json, write_jsonl
from .metrics import binary_metrics, permutation_metrics
from .schema import PatientRecord
from .serialize import (
    ClinTimeCompactConfig,
    ClinTimeFeatureSummaryConfig,
    ClinTimeJSONConfig,
    ClinTimeXMLConfig,
    MEDSConfig,
    SerializationConfig,
    serialize_clintime_compact,
    serialize_clintime_feature_summary,
    serialize_clintime_json,
    serialize_clintime_xml,
    serialize_events,
    serialize_meds_events,
    stable_view_seed,
)


DEFAULT_TASK_PROMPT = (
    "Task: predict in-hospital mortality from observations recorded during the first 6 ICU hours. "
    "Use only the supplied events. Return a mortality risk score.\n"
)
SYSTEM_PREFIX = DEFAULT_TASK_PROMPT

XML_VARIANTS = {"xml", "xml_aug", "xml_kl"}
COMPACT_VARIANTS = {"compact", "compact_aug", "compact_kl"}
JSON_VARIANTS = {"json", "json_aug", "json_kl"}
SUMMARY_VARIANTS = {"summary", "summary_wide"}
TWO_VIEW_VARIANTS = {"xml_aug", "xml_kl", "compact_aug", "compact_kl", "json_aug", "json_kl"}


def _uses_two_views(variant: str) -> bool:
    return variant in TWO_VIEW_VARIANTS


def _inference_variant(variant: str) -> str:
    if variant in XML_VARIANTS:
        return "xml"
    if variant in COMPACT_VARIANTS:
        return "compact"
    if variant in JSON_VARIANTS:
        return "json"
    if variant in SUMMARY_VARIANTS:
        return "summary"
    return variant


def _records(path: str, split: str, limit: int | None, seed: int, balance: bool = False) -> list[PatientRecord]:
    """Deterministic class-stratified sampling, optionally balanced for training."""
    values = None
    for attempt in range(3):
        try:
            values = [PatientRecord.from_dict(value) for value in read_jsonl(path) if value.get("split") == split]
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    assert values is not None
    if limit is None and not balance:
        return values
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for value in values:
        by_label[value.label].append(value)
    if balance:
        if limit is None:
            per_class = min(len(by_label[0]), len(by_label[1]))
        else:
            per_class = max(1, limit // 2)
        chosen = []
        for label in (1, 0):
            pool = by_label[label]
            if not pool:
                continue
            if len(pool) >= per_class:
                chosen.extend(rng.sample(pool, per_class))
            else:
                chosen.extend(pool)
                chosen.extend(rng.choices(pool, k=per_class - len(pool)))
        rng.shuffle(chosen)
        return chosen
    if limit is None or len(values) <= limit:
        return values
    n_pos = round(limit * len(by_label[1]) / len(values))
    chosen = rng.sample(by_label[1], min(n_pos, len(by_label[1])))
    chosen += rng.sample(by_label[0], min(limit - len(chosen), len(by_label[0])))
    rng.shuffle(chosen)
    return chosen


def _text(
    record: PatientRecord,
    variant: str,
    seed: int,
    second_view: bool = False,
    force_permutation: bool | None = None,
    task_prompt: str = DEFAULT_TASK_PROMPT,
) -> str:
    view_seed = stable_view_seed(record.patient_id, seed + (1_000_003 if second_view else 0))
    if variant == "flat":
        permute = bool(force_permutation) if force_permutation is not None else False
        body = serialize_events(
            record.events or [],
            SerializationConfig(permute_within_sets=permute, max_events=128, max_chars=12_000),
            view_seed,
        )
    elif variant == "meds":
        body = serialize_meds_events(record.events or [], MEDSConfig(max_events=160, max_chars=6_000))
    elif variant in COMPACT_VARIANTS:
        permute = _uses_two_views(variant) if force_permutation is None else force_permutation
        body = serialize_clintime_compact(
            record.events or [],
            ClinTimeCompactConfig(permute_within_blocks=permute, max_nodes=160, max_chars=6_000),
            view_seed,
        )
    elif variant in JSON_VARIANTS:
        permute = _uses_two_views(variant) if force_permutation is None else force_permutation
        body = serialize_clintime_json(
            record.events or [],
            ClinTimeJSONConfig(permute_within_blocks=permute, max_nodes=160, max_chars=6_000),
            view_seed,
        )
    elif variant in SUMMARY_VARIANTS:
        summary_config = (
            ClinTimeFeatureSummaryConfig(max_rows=256, max_chars=6_000)
            if variant == "summary_wide"
            else ClinTimeFeatureSummaryConfig(max_rows=80, max_chars=3_000)
        )
        body = serialize_clintime_feature_summary(
            record.events or [],
            summary_config,
        )
    else:
        permute = _uses_two_views(variant) if force_permutation is None else force_permutation
        body = serialize_clintime_xml(
            record.events or [],
            ClinTimeXMLConfig(permute_within_blocks=permute, max_nodes=128, max_chars=12_000),
            view_seed,
        )
    return task_prompt.rstrip() + "\n" + body


def _symmetric_bernoulli_kl(logits_a, logits_b):
    import torch

    p = torch.sigmoid(logits_a.float()).clamp(1e-6, 1 - 1e-6)
    q = torch.sigmoid(logits_b.float()).clamp(1e-6, 1 - 1e-6)
    kl_pq = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
    kl_qp = q * torch.log(q / p) + (1 - q) * torch.log((1 - q) / (1 - p))
    return 0.5 * (kl_pq + kl_qp).mean()


def _model_layout(model) -> dict[str, list[str]]:
    """Resolve equivalent classifier heads and attention projections across backbones."""
    modules = dict(model.named_modules())
    head_names = [name for name in ("classifier_head", "score", "classifier", "classification_head") if name in modules]
    if not head_names:
        head_names = [
            name for name, module in modules.items()
            if name and getattr(module, "out_features", None) == 1
            and name.rsplit(".", 1)[-1] in {"score", "classifier", "classifier_head", "classification_head"}
        ]
    suffixes = {name.rsplit(".", 1)[-1] for name in modules}
    if "query_key_value" in suffixes:
        targets = ["query_key_value"]
    else:
        targets = [name for name in ("q_proj", "k_proj", "v_proj", "o_proj") if name in suffixes]
    if not head_names:
        raise RuntimeError("Could not locate a binary classification head")
    if not targets:
        raise RuntimeError("Could not locate attention projection modules for LoRA")
    return {"head_names": head_names, "lora_targets": targets}


def _initialize_classification_heads(model, head_names: list[str]) -> None:
    """Initialize downstream heads deterministically, including ChatGLM empty-init heads."""
    import torch

    modules = dict(model.named_modules())
    for name in head_names:
        module = modules[name]
        for parameter_name, parameter in module.named_parameters(recurse=True):
            if parameter.ndim >= 2 and parameter_name.endswith("weight"):
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
            else:
                torch.nn.init.zeros_(parameter)


def _load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    except Exception as fast_error:
        try:
            return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        except Exception as slow_error:
            raise RuntimeError(
                f"Could not load either fast or slow tokenizer for {model_name}: "
                f"fast={fast_error!r}; slow={slow_error!r}"
            ) from slow_error


def _load_base(model_name: str, training: bool):
    import torch
    from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

    tokenizer = _load_tokenizer(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_4bit = os.environ.get("CLINTIME_NO_4BIT", "0") != "1"
    quant = None
    if use_4bit:
        try:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        except Exception:
            use_4bit = False
            quant = None
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "num_labels": 1,
        "torch_dtype": torch.bfloat16,
    }
    if quant is not None:
        model_kwargs["quantization_config"] = quant
        model_kwargs["device_map"] = {"": 0}
        # Transformers 4.44 + Accelerate 1.14 can otherwise route recent bnb
        # models through a single-device `.to()` call, which bnb forbids.
        import transformers.modeling_utils as modeling_utils

        original_dispatch_model = modeling_utils.dispatch_model

        def _dispatch_model_with_hooks(*args: Any, **kwargs: Any):
            kwargs["force_hooks"] = True
            return original_dispatch_model(*args, **kwargs)

        modeling_utils.dispatch_model = _dispatch_model_with_hooks
    else:
        original_dispatch_model = None
        model_kwargs["device_map"] = {"": "cuda:0"}
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **model_kwargs,
        )
    finally:
        if original_dispatch_model is not None:
            modeling_utils.dispatch_model = original_dispatch_model
    layout = _model_layout(model)
    # ChatGLM's remote-code sequence classifier can be absent and left uninitialized.
    # Reinitializing every downstream head also keeps backbone comparisons reproducible.
    _initialize_classification_heads(model, layout["head_names"])
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if training:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

        if use_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        elif hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_CLS,
                target_modules=layout["lora_targets"],
                modules_to_save=list(layout["head_names"]),
            ),
        )
    return tokenizer, model, layout


def _restore_saved_classification_heads(model, adapter_dir: Path) -> list[str]:
    """Force saved sequence-classification heads onto PEFT-wrapped backbones.

    ChatGLM remote-code classifiers expose a `classifier_head` module that PEFT
    records via `modules_to_save`, but reloads can leave the forward path using a
    freshly initialized head.  We explicitly copy any saved `score`/`classifier`
    tensors into the live module after `PeftModel.from_pretrained`.
    """
    adapter_path = adapter_dir / "adapter_model.safetensors"
    if not adapter_path.exists():
        return []
    try:
        import torch
        from safetensors.torch import load_file
    except Exception:
        return []

    state = load_file(str(adapter_path), device="cpu")
    restored: list[str] = []
    modules = dict(model.named_modules())
    active_adapter = getattr(model, "active_adapter", "default")
    if callable(active_adapter):
        active_adapter = active_adapter()
    for key, tensor in state.items():
        if not (key.endswith(".weight") or key.endswith(".bias")):
            continue
        if not any(name in key for name in ("classifier_head", "classification_head", "classifier", "score")):
            continue
        module_name, parameter_name = key.rsplit(".", 1)
        module = modules.get(module_name)
        if module is None:
            # PEFT may expose modules with or without the leading `base_model.model`.
            suffix = module_name.split(".", 2)[-1]
            matches = [value for name, value in modules.items() if name == suffix or name.endswith("." + suffix)]
            module = matches[0] if matches else None
        if module is None:
            continue
        target = module
        modules_to_save = getattr(module, "modules_to_save", None)
        if modules_to_save is not None:
            if active_adapter in modules_to_save:
                target = modules_to_save[active_adapter]
            elif len(modules_to_save):
                target = next(iter(modules_to_save.values()))
        parameter = getattr(target, parameter_name, None)
        if parameter is None or tuple(parameter.shape) != tuple(tensor.shape):
            continue
        with torch.no_grad():
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
        restored.append(key)
    return restored


def _load_adapter(model, checkpoint: Path):
    from peft import PeftModel

    adapter_dir = checkpoint / "adapter"
    model = PeftModel.from_pretrained(model, adapter_dir)
    restored = _restore_saved_classification_heads(model, adapter_dir)
    if restored:
        print(json.dumps({"restored_classification_heads": restored}), flush=True)
    return model


def _batch(tokenizer, texts: list[str], max_length: int):
    return {
        key: value.to("cuda")
        for key, value in tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").items()
    }


def _predict(
    model,
    tokenizer,
    records: list[PatientRecord],
    variant: str,
    seed: int,
    max_length: int,
    task_prompt: str = DEFAULT_TASK_PROMPT,
) -> np.ndarray:
    import torch

    model.eval()
    result: list[float] = []
    with torch.inference_mode():
        for record in records:
            # All XML variants use one canonical, non-permuted representation at inference.
            inference_variant = _inference_variant(variant)
            logits = model(
                **_batch(tokenizer, [_text(record, inference_variant, seed, task_prompt=task_prompt)], max_length)
            ).logits.view(-1)
            result.append(float(torch.sigmoid(logits.float())[0].cpu()))
    return np.asarray(result)


def _predict_permutations(
    model,
    tokenizer,
    records: list[PatientRecord],
    variant: str,
    seed: int,
    max_length: int,
    permutations: int,
    view_batch_size: int,
    task_prompt: str = DEFAULT_TASK_PROMPT,
) -> np.ndarray:
    import torch

    if permutations < 2:
        raise ValueError("permutations must be at least 2")
    model.eval()
    rows: list[list[float]] = []
    with torch.inference_mode():
        for record in records:
            texts = [
                _text(record, variant, seed + 10_007 * index, force_permutation=True, task_prompt=task_prompt)
                for index in range(permutations)
            ]
            probabilities: list[float] = []
            for start in range(0, len(texts), max(1, view_batch_size)):
                logits = model(**_batch(tokenizer, texts[start:start + max(1, view_batch_size)], max_length)).logits.view(-1)
                probabilities.extend(torch.sigmoid(logits.float()).detach().cpu().tolist())
            rows.append(probabilities)
    return np.asarray(rows, dtype=float)


def _hospital_metrics(records: list[PatientRecord], probabilities: np.ndarray) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        if record.hospital_id:
            groups.setdefault(record.hospital_id, []).append(index)
    per_hospital: dict[str, Any] = {}
    for hospital, indices in groups.items():
        y = [records[index].label for index in indices]
        p = [probabilities[index] for index in indices]
        per_hospital[hospital] = binary_metrics(y, p)
    aurocs = [value["auroc"] for value in per_hospital.values() if math.isfinite(value["auroc"])]
    auprcs = [value["auprc"] for value in per_hospital.values() if math.isfinite(value["auprc"])]
    return {
        "hospital_count": len(groups),
        "macro_auroc": float(np.mean(aurocs)) if aurocs else math.nan,
        "macro_auprc": float(np.mean(auprcs)) if auprcs else math.nan,
        "per_hospital": per_hospital,
    }


def train(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F

    seed_everything(args.seed)
    train_records = _records(args.data, "train", args.max_train_records, args.sample_seed, args.balance_train)
    val_records = _records(args.data, "val", args.max_val_records, args.sample_seed + 1)
    tokenizer, model, layout = _load_base(args.model, training=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    model.train()
    steps = 0
    losses: list[float] = []
    step_losses: list[float] = []
    positives = sum(record.label for record in train_records)
    negatives = len(train_records) - positives
    if args.pos_weight is not None:
        pos_weight_value = float(args.pos_weight)
    elif args.auto_pos_weight and positives > 0:
        pos_weight_value = float(negatives / positives)
    else:
        pos_weight_value = 1.0
    if args.pos_weight_cap is not None:
        pos_weight_value = min(pos_weight_value, float(args.pos_weight_cap))
    pos_weight = torch.tensor([pos_weight_value], device="cuda") if pos_weight_value != 1.0 else None

    def supervised_loss(logits, target):
        return F.binary_cross_entropy_with_logits(logits.float(), target, pos_weight=pos_weight)

    for epoch in range(args.epochs):
        order = list(range(len(train_records)))
        random.Random(args.seed + epoch).shuffle(order)
        for position, index in enumerate(order, 1):
            record = train_records[index]
            logits_a = model(
                **_batch(
                    tokenizer,
                    [_text(record, args.variant, args.seed + epoch, task_prompt=args.task_prompt)],
                    args.max_length,
                )
            ).logits.view(-1)
            target = torch.tensor([float(record.label)], device="cuda")
            if _uses_two_views(args.variant):
                logits_b = model(
                    **_batch(
                        tokenizer,
                        [_text(record, args.variant, args.seed + epoch, True, task_prompt=args.task_prompt)],
                        args.max_length,
                    )
                ).logits.view(-1)
                supervised = 0.5 * (supervised_loss(logits_a, target) + supervised_loss(logits_b, target))
                loss = supervised
                if args.variant in {"xml_kl", "compact_kl", "json_kl"}:
                    loss = loss + args.kl_weight * _symmetric_bernoulli_kl(logits_a, logits_b)
            else:
                loss = supervised_loss(logits_a, target)
            (loss / args.grad_accum).backward()
            losses.append(float(loss.detach().cpu()))
            if position % args.grad_accum == 0 or position == len(order):
                recent_micro_losses = losses[-min(args.grad_accum, len(losses)) :]
                step_loss = float(np.mean(recent_micro_losses)) if recent_micro_losses else float(loss.detach().cpu())
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                step_losses.append(step_loss)
                log_entry = {"epoch": epoch + 1, "step": steps, "loss": round(float(loss), 6)}
                print(json.dumps(log_entry), flush=True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")
    probabilities = _predict(model, tokenizer, val_records, args.variant, args.seed, args.max_length, args.task_prompt)
    metrics = binary_metrics([record.label for record in val_records], probabilities)
    train_eval_metrics = None
    if args.train_eval_records:
        train_eval_records = _records(args.data, "train", args.train_eval_records, args.sample_seed)
        train_eval_probabilities = _predict(
            model, tokenizer, train_eval_records, args.variant, args.seed, args.max_length, args.task_prompt
        )
        train_eval_metrics = binary_metrics([record.label for record in train_eval_records], train_eval_probabilities)
    metadata = {
        "variant": args.variant,
        "seed": args.seed,
        "sample_seed": args.sample_seed,
        "base_model": args.model,
        "max_length": args.max_length,
        "kl_weight": args.kl_weight,
        "task_prompt": args.task_prompt,
        "pos_weight": pos_weight_value,
        "pos_weight_cap": args.pos_weight_cap,
        "auto_pos_weight": args.auto_pos_weight,
        "balance_train": args.balance_train,
        "train_positive_rate": float(positives / len(train_records)) if train_records else math.nan,
        "train_records": len(train_records),
        "val_records": len(val_records),
        "optimizer_steps": steps,
        "mean_train_loss": float(np.mean(losses)),
        "mean_optimizer_step_loss": float(np.mean(step_losses)) if step_losses else math.nan,
        "elapsed_seconds": time.time() - started,
        "max_gpu_memory_mb": torch.cuda.max_memory_allocated() // 2**20,
        "model_layout": layout,
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "validation": metrics,
    }
    if train_eval_metrics is not None:
        metadata["train_eval"] = train_eval_metrics
    write_json(out / "train_metrics.json", metadata)
    write_json(out / "run_config.json", {key: value for key, value in vars(args).items() if key != "func"})
    for specification in args.eval_data:
        dataset, data_path = specification.split("=", 1)
        evaluation_records = _records(data_path, "test", args.max_eval_records, args.sample_seed + 2)
        evaluation_probabilities = _predict(
            model, tokenizer, evaluation_records, args.variant, args.seed, args.max_length, args.task_prompt
        )
        evaluation_metrics = binary_metrics(
            [record.label for record in evaluation_records], evaluation_probabilities
        )
        evaluation_metrics.update({"dataset": dataset, "split": "test", "variant": args.variant, "seed": args.seed})
        if dataset == "eicu":
            evaluation_metrics["hospital"] = _hospital_metrics(evaluation_records, evaluation_probabilities)
        evaluation_output = out / "evaluation" / dataset
        write_json(evaluation_output / "metrics.json", evaluation_metrics)
        write_jsonl(
            evaluation_output / "predictions.jsonl",
            ({"patient_id": record.patient_id, "stay_id": record.stay_id, "label": record.label,
              "probability": float(probability)}
             for record, probability in zip(evaluation_records, evaluation_probabilities)),
        )
        print(json.dumps({"evaluation": evaluation_metrics}), flush=True)
    print(json.dumps(metadata), flush=True)


def evaluate(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    run_config = json.loads((checkpoint / "run_config.json").read_text(encoding="utf-8"))
    variant, seed = run_config["variant"], int(run_config["seed"])
    sample_seed = int(run_config.get("sample_seed", seed))
    task_prompt = run_config.get("task_prompt", DEFAULT_TASK_PROMPT)
    records = _records(args.data, args.split, args.max_eval_records, sample_seed + 2)
    seed_everything(seed)
    tokenizer, model, _ = _load_base(run_config["model"], training=False)
    model = _load_adapter(model, checkpoint)
    probabilities = _predict(model, tokenizer, records, variant, seed, int(run_config["max_length"]), task_prompt)
    metrics = binary_metrics([record.label for record in records], probabilities)
    metrics.update({"dataset": args.dataset, "split": args.split, "variant": variant, "seed": seed})
    if args.dataset == "eicu":
        metrics["hospital"] = _hospital_metrics(records, probabilities)
    output = Path(args.output)
    write_json(output / "metrics.json", metrics)
    write_jsonl(
        output / "predictions.jsonl",
        ({"patient_id": record.patient_id, "stay_id": record.stay_id, "label": record.label, "probability": float(probability)}
         for record, probability in zip(records, probabilities)),
    )
    print(json.dumps(metrics), flush=True)


def audit_permutation(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    run_config = json.loads((checkpoint / "run_config.json").read_text(encoding="utf-8"))
    variant, seed = run_config["variant"], int(run_config["seed"])
    sample_seed = int(run_config.get("sample_seed", seed))
    task_prompt = run_config.get("task_prompt", DEFAULT_TASK_PROMPT)
    records = _records(args.data, args.split, args.max_records, sample_seed + 2)
    seed_everything(seed)
    tokenizer, model, _ = _load_base(run_config["model"], training=False)
    model = _load_adapter(model, checkpoint)
    max_length = int(run_config["max_length"])
    canonical = _predict(model, tokenizer, records, variant, seed, max_length, task_prompt)
    matrix = _predict_permutations(
        model,
        tokenizer,
        records,
        variant,
        args.audit_seed,
        max_length,
        args.permutations,
        args.view_batch_size,
        task_prompt,
    )
    labels = [record.label for record in records]
    result = {
        "dataset": args.dataset,
        "split": args.split,
        "variant": variant,
        "seed": seed,
        "audit_seed": args.audit_seed,
        "n": len(records),
        "canonical": binary_metrics(labels, canonical),
        "permutation_ensemble": binary_metrics(labels, matrix.mean(axis=1)),
        "permutation": permutation_metrics(matrix),
        "mean_abs_canonical_ensemble_gap": float(np.abs(canonical - matrix.mean(axis=1)).mean()),
    }
    output = Path(args.output)
    write_json(output / "metrics.json", result)
    np.save(output / "permutation_probabilities.npy", matrix)
    write_jsonl(
        output / "predictions.jsonl",
        (
            {
                "patient_id": record.patient_id,
                "stay_id": record.stay_id,
                "label": record.label,
                "canonical_probability": float(canonical[index]),
                "permutation_probabilities": [float(value) for value in matrix[index]],
            }
            for index, record in enumerate(records)
        ),
    )
    print(json.dumps(result), flush=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    sub = value.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--variant",
        choices=[
            "flat", "xml", "xml_aug", "xml_kl",
            "compact", "compact_aug", "compact_kl",
            "json", "json_aug", "json_kl",
            "summary", "summary_wide", "meds",
        ],
        required=True,
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-seed", type=int, default=2026)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--kl-weight", type=float, default=0.2)
    p.add_argument("--task-prompt", default=DEFAULT_TASK_PROMPT)
    p.add_argument("--pos-weight", type=float)
    p.add_argument("--pos-weight-cap", type=float)
    p.add_argument("--auto-pos-weight", action="store_true")
    p.add_argument("--balance-train", action="store_true")
    p.add_argument("--train-eval-records", type=int)
    p.add_argument("--max-train-records", type=int)
    p.add_argument("--max-val-records", type=int)
    p.add_argument("--eval-data", action="append", default=[], metavar="DATASET=PATH")
    p.add_argument("--max-eval-records", type=int)
    p.set_defaults(func=train)
    p = sub.add_parser("eval")
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", required=True, choices=["mimic", "eicu", "nwicu"])
    p.add_argument("--split", default="test")
    p.add_argument("--max-eval-records", type=int)
    p.set_defaults(func=evaluate)
    p = sub.add_parser("audit-permutation")
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", required=True, choices=["mimic", "eicu", "nwicu"])
    p.add_argument("--split", default="test")
    p.add_argument("--max-records", type=int, default=500)
    p.add_argument("--permutations", type=int, default=5)
    p.add_argument("--audit-seed", type=int, default=2026)
    p.add_argument("--view-batch-size", type=int, default=2)
    p.set_defaults(func=audit_permutation)
    return value


def main() -> None:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
