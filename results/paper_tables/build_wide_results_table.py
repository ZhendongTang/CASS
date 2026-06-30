import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_draft" / "tables" / "wide_results_summary.tex"
ASSETS = json.loads((ROOT / "paper_draft" / "figures" / "summary_assets_20260630.json").read_text(encoding="utf-8"))
REMOTE = ROOT / "refine-logs" / "remote-results"


def f3(x):
    if x is None:
        return "--"
    return f"{x:.3f}".lstrip("0")


def cell(auroc=None, auprc=None):
    return f"{f3(auroc)}/{f3(auprc)}" if auprc is not None else f"{f3(auroc)}/--"


def mean_cell(values):
    values = [v for v in values if v is not None]
    if not values:
        return "--"
    if len(values) == 1:
        return f3(values[0])
    return f"{f3(mean(values))}$\\pm${f3(pstdev(values))}"


def pair_mean_cell(rows):
    aurocs = [r.get("auroc") for r in rows if r.get("auroc") is not None]
    auprcs = [r.get("auprc") for r in rows if r.get("auprc") is not None]
    if not aurocs:
        return "--"
    return f"{mean_cell(aurocs)}/{mean_cell(auprcs)}" if auprcs else f"{mean_cell(aurocs)}/--"


def avg_token(format_name):
    key = format_name.lower()
    if key == "cass":
        key = "summary"
    token = ASSETS["token_audit"]
    vals = []
    trunc = []
    for center in ("mimic", "eicu", "nwicu"):
        if key in token[center]:
            vals.append(token[center][key]["mean_tokens"])
            trunc.append(token[center][key]["truncation_rate_at_2048"] * 100)
    if not vals:
        return "--"
    return f"{mean(vals):.0f}/{mean(trunc):.1f}\\%"


def add_row(rows, block, fmt, task, model, train, ctx, seeds, centers, token="--"):
    rows.append({
        "block": block,
        "format": fmt,
        "task": task,
        "model": model,
        "train": train,
        "ctx": ctx,
        "seeds": seeds,
        "mimic": centers.get("MIMIC-IV", "--"),
        "eicu": centers.get("eICU", "--"),
        "nwicu": centers.get("NWICU", "--"),
        "token": token,
    })


rows = []

# Matched 2048-token mortality format comparison.
fmt_rows = defaultdict(list)
for r in ASSETS["formats"]:
    fmt_rows[(r["format"], r["center"])].append(r)
for fmt in ("Summary", "Compact", "XML", "Flat"):
    centers = {}
    for center in ("MIMIC-IV", "eICU", "NWICU"):
        rs = fmt_rows.get((fmt, center), [])
        if rs:
            centers[center] = pair_mean_cell(rs)
    add_row(
        rows,
        "Matched 2048",
        "CASS" if fmt == "Summary" else fmt,
        "Mortality",
        "Llama-3-8B",
        "5k",
        "2048",
        "42",
        centers,
        avg_token("CASS" if fmt == "Summary" else fmt),
    )

# Scaled CASS mortality.
mort = defaultdict(list)
for r in ASSETS["mortality"]:
    mort[(r["train"], r["center"])].append(r)
for train, seeds in (("5k", "42--44"), ("10k", "42"), ("20k", "42--44")):
    centers = {}
    for center in ("MIMIC-IV", "eICU", "NWICU"):
        centers[center] = pair_mean_cell(mort.get((train, center), []))
    add_row(rows, "Scaled CASS", "CASS", "Mortality", "Llama-3-8B", train, "2048", seeds, centers, avg_token("CASS"))

# CASS multitask rows.
multi = defaultdict(list)
for r in ASSETS["multitask"]:
    multi[(r["task"], r["train"], r["center"])].append(r)
for task, train, seeds in [
    ("LOS >48h", "5k", "42"),
    ("LOS >48h", "10k", "42"),
    ("Future intervention", "5k", "42"),
    ("Future intervention", "10k", "42--44"),
]:
    centers = {}
    for center in ("MIMIC-IV", "eICU", "NWICU"):
        centers[center] = pair_mean_cell(multi.get((task, train, center), []))
    add_row(rows, "Multitask CASS", "CASS", task, "Llama-3-8B", train, "2048", seeds, centers, avg_token("CASS"))

# Early 1024-token pilots.
pilot_specs = [
    ("Compact-KL", REMOTE / "final_v1" / "compact_pilot" / "llama_compact_kl_s42_n1000_l1024"),
    ("JSON-KL", REMOTE / "final_v1" / "json_pilot" / "llama_json_kl_s42_n1000_l1024"),
]
for fmt, root in pilot_specs:
    centers = {}
    for center_key, label in [("mimic", "MIMIC-IV"), ("eicu", "eICU"), ("nwicu", "NWICU")]:
        p = root / "evaluation" / center_key / "metrics.json"
        if p.exists():
            r = json.loads(p.read_text(encoding="utf-8"))
            centers[label] = cell(r.get("auroc"), r.get("auprc"))
    add_row(rows, "1024 pilot", fmt, "Mortality", "Llama-3-8B", "1k", "1024", "42", centers, "--")

# Early 256-token multi-model diagnostics.
model_labels = {"glm": "GLM-4-9B", "llama": "Llama-3-8B", "gemma": "Gemma-2-9B"}
variant_labels = {"flat": "Flat", "xml": "XML", "xml_aug": "XML-Aug", "xml_kl": "XML+KL"}
for model_key, model_label in model_labels.items():
    for variant_key, variant_label in variant_labels.items():
        centers = {}
        for center_key, center_label in [("mimic", "MIMIC-IV"), ("eicu", "eICU"), ("nwicu", "NWICU")]:
            rs = []
            for seed in (42, 43, 44):
                p = REMOTE / "final_v1" / "main" / model_key / f"{variant_key}_seed{seed}" / "evaluation" / center_key / "metrics.json"
                if p.exists():
                    rs.append(json.loads(p.read_text(encoding="utf-8")))
            centers[center_label] = pair_mean_cell(rs)
        add_row(rows, "256 diagnostic", variant_label, "Mortality", model_label, "1k", "256", "42--44", centers, "--")

lines = []
lines.append(r"\begin{table*}[t]")
lines.append(r"\caption{Wide summary of completed LLM experiments across serialization formats, tasks, backbones, and centers. Cells report AUROC/AUPRC; values with $\pm$ are mean$\pm$standard deviation across completed seeds. The 2048-token blocks are the main paper evidence, while the 1024- and 256-token blocks are earlier diagnostic runs with smaller training budgets. Token/truncation reports mean tokens and 2048-token truncation rate averaged over the three-center token audit.}")
lines.append(r"\label{tab:wide_results}")
lines.append(r"\centering")
lines.append(r"\scriptsize")
lines.append(r"\setlength{\tabcolsep}{2.2pt}")
lines.append(r"\resizebox{\textwidth}{!}{%")
lines.append(r"\begin{tabular}{lllccclcccc}")
lines.append(r"\toprule")
lines.append(r"Block & Format & Task & Model & Train & Ctx & Seeds & MIMIC-IV & eICU & NWICU & Tok./trunc. \\")
lines.append(r"\midrule")
last_block = None
for r in rows:
    if last_block is not None and r["block"] != last_block:
        lines.append(r"\midrule")
    last_block = r["block"]
    vals = [r["block"], r["format"], r["task"], r["model"], r["train"], r["ctx"], r["seeds"], r["mimic"], r["eicu"], r["nwicu"], r["token"]]
    lines.append(" & ".join(vals) + r" \\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}%")
lines.append(r"}")
lines.append(r"\end{table*}")
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(OUT)
