"""Aggregate results/runs/*.json into results/summary.json, results/RESULTS.md and results/figures/*.png.

Every number in the README is produced by this script; nothing is copied by hand.

Example:
    python scripts/make_report.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from lm_distill.utils import load_json, save_json  # noqa: E402

DISTILLED, BASELINE, TEACHER, INK, MUTED = "#2563eb", "#ea580c", "#059669", "#111827", "#6b7280"
METRICS = ("perplexity", "kl", "agreement")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9ca3af",
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e5e7eb", "grid.linewidth": 0.8,
    "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False,
})
plain = FuncFormatter(lambda v, _: f"{v:g}")


def log_ticks(ax, candidates=(2, 5, 10, 20, 50, 100, 200, 500, 1000)) -> None:
    """Readable tick labels on a log y-axis (matplotlib labels only powers of ten by default)."""
    lo, hi = ax.get_ylim()
    ax.set_yticks([t for t in candidates if lo <= t <= hi])
    ax.yaxis.set_major_formatter(plain)
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


# ----------------------------------------------------------------------------- aggregation
def group_key(run: dict) -> tuple:
    c = run["config"]
    t = c["train"]
    distill = t["method"] == "distill"
    return (t["method"], c["data"]["num_train_stories"], t["temperature"] if distill else None,
            t["alpha"] if distill else None)


def aggregate(runs: list[dict]) -> dict[tuple, dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        groups[group_key(r)].append(r)
    out = {}
    for key, rs in groups.items():
        rs.sort(key=lambda r: r["config"]["train"]["seed"])
        entry = {"seeds": [r["config"]["train"]["seed"] for r in rs], "runs": [r["run_name"] for r in rs]}
        for m in METRICS:
            vals = np.array([r["final"][m] for r in rs])
            entry[m] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                        "values": vals.tolist()}
        steps = [h["step"] for h in rs[0]["history"]]
        curves = np.array([[h["perplexity"] for h in r["history"]] for r in rs])
        entry["curve"] = {"step": steps, "mean": curves.mean(0).tolist(), "std": curves.std(0, ddof=1).tolist()
                          if len(rs) > 1 else [0.0] * len(steps)}
        entry["train_seconds"] = float(np.mean([r["train_seconds"] for r in rs]))
        out[key] = entry
    return out


def get(agg, method, n=30000, temperature=1.0, alpha=1.0):
    key = (method, n, temperature, alpha) if method == "distill" else (method, n, None, None)
    return agg.get(key)


def fmt(entry, metric="perplexity", pct=False, digits=2):
    if entry is None:
        return "–"
    m, s = entry[metric]["mean"], entry[metric]["std"]
    return f"{m:.1%} ± {s:.1%}" if pct else f"{m:.{digits}f} ± {s:.{digits}f}"


# ----------------------------------------------------------------------------- figures
def fig_soft_labels(soft: dict, path: Path) -> None:
    rows = soft["top"][::-1]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bars = ax.barh([repr(r["token"]) for r in rows], [100 * r["prob"] for r in rows], color=TEACHER, height=0.62)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9, color=INK)
    ax.set(xlabel="Teacher probability (%)", title=f'Teacher soft label after "{soft["prompt"]}"')
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 100 * soft["top"][0]["prob"] * 1.15)
    fig.savefig(path)
    plt.close(fig)


def _curve(ax, entry, color, label):
    x, m, s = np.array(entry["curve"]["step"]), np.array(entry["curve"]["mean"]), np.array(entry["curve"]["std"])
    ax.plot(x, m, "o-", color=color, label=label, markeredgecolor="white", markeredgewidth=1.2)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, linewidth=0)


def fig_training_curves(agg, teacher, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, n, title in [(axes[0], 30000, "30,000 training stories"), (axes[1], 300, "300 training stories")]:
        _curve(ax, get(agg, "baseline", n), BASELINE, "Baseline (cross-entropy)")
        _curve(ax, get(agg, "distill", n), DISTILLED, "Distilled (teacher soft labels)")
        ax.axhline(teacher["perplexity"], color=TEACHER, ls="--", lw=1.6,
                   label=f"Teacher ({teacher['perplexity']:.2f})")
        ax.set(yscale="log", xlabel="Training step", title=title, xticks=get(agg, "baseline", n)["curve"]["step"])
        log_ticks(ax)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Validation perplexity (log)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.text(1.0, -0.02, "Mean ± 1 std over 3 seeds", ha="right", fontsize=8, color=MUTED)
    fig.savefig(path)
    plt.close(fig)


def fig_data_scarcity(agg, teacher, path: Path) -> None:
    sizes = sorted({k[1] for k in agg})
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for method, color, label in [("baseline", BASELINE, "Baseline"), ("distill", DISTILLED, "Distilled")]:
        es = [get(agg, method, n) for n in sizes]
        ax.errorbar(sizes, [e["perplexity"]["mean"] for e in es], yerr=[e["perplexity"]["std"] for e in es],
                    fmt="o-", color=color, label=label, capsize=3, markeredgecolor="white", markeredgewidth=1.2)
    ax.axhline(teacher["perplexity"], color=TEACHER, ls="--", lw=1.6, label="Teacher")
    for n in sizes:
        b, d = get(agg, "baseline", n)["perplexity"]["mean"], get(agg, "distill", n)["perplexity"]["mean"]
        ax.annotate(f"{b / d:.1f}× lower", (n, d), ha="center", va="top", xytext=(0, -9),
                    textcoords="offset points", fontsize=9, fontweight="bold", color=DISTILLED)
    ax.set(xscale="log", yscale="log", xlabel="Training stories", ylabel="Final validation perplexity (log)",
           title="Distillation helps most when text is scarce", xticks=sizes)
    ax.set_xlim(sizes[0] / 1.6, sizes[-1] * 1.6)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    log_ticks(ax)
    ax.legend(loc="upper right")
    fig.savefig(path)
    plt.close(fig)


def fig_temperature(agg, path: Path) -> None:
    temps = sorted(k[2] for k in agg if k[0] == "distill" and k[1] == 30000 and k[3] == 1.0)
    es = [get(agg, "distill", 30000, t) for t in temps]
    base = get(agg, "baseline", 30000)["perplexity"]["mean"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar([f"T = {t:g}" for t in temps], [e["perplexity"]["mean"] for e in es],
           yerr=[e["perplexity"]["std"] for e in es], color=DISTILLED, width=0.55, capsize=4,
           error_kw={"ecolor": INK, "elinewidth": 1.2, "capthick": 1.2})
    ax.axhline(base, color=BASELINE, ls="--", lw=1.6, label=f"Baseline ({base:.2f})")
    ax.set(ylabel="Final validation perplexity", title="Temperature ablation (α = 1)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- markdown
def write_markdown(agg, teacher, bench, path: Path) -> None:
    b, d = get(agg, "baseline"), get(agg, "distill")
    lines = ["# Results", "", "Generated by `scripts/make_report.py`. Mean ± std over seeds "
             f"{d['seeds']}; validation = {1000} held-out chunks of 64 tokens.", "",
             "## Main comparison (30,000 training stories)", "",
             "| Model | Params | Perplexity ↓ | KL to teacher ↓ | Top-1 agreement ↑ |", "|---|---|---|---|---|",
             f"| Teacher (TinyStories-33M) | {teacher['params'] / 1e6:.1f}M | {teacher['perplexity']:.2f} | 0 | 100% |",
             f"| Baseline student | 6.8M | {fmt(b)} | {fmt(b, 'kl', digits=3)} | {fmt(b, 'agreement', True)} |",
             f"| **Distilled student** | 6.8M | **{fmt(d)}** | **{fmt(d, 'kl', digits=3)}** | "
             f"**{fmt(d, 'agreement', True)}** |", "",
             "## Training-data scarcity", "", "| Training stories | Baseline | Distilled | Baseline / Distilled |",
             "|---|---|---|---|"]
    for n in sorted({k[1] for k in agg}):
        bn, dn = get(agg, "baseline", n), get(agg, "distill", n)
        lines.append(f"| {n:,} | {fmt(bn)} | {fmt(dn)} | {bn['perplexity']['mean'] / dn['perplexity']['mean']:.1f}× |")
    lines += ["", "## Temperature ablation (30,000 stories, α = 1)", "", "| Temperature | Perplexity | KL to teacher |",
              "|---|---|---|"]
    for t in sorted(k[2] for k in agg if k[0] == "distill" and k[1] == 30000 and k[3] == 1.0):
        e = get(agg, "distill", 30000, t)
        lines.append(f"| {t:g} | {fmt(e)} | {fmt(e, 'kl', digits=3)} |")
    if bench:
        t, s, r = bench["teacher"], bench["student"], bench["ratios"]
        lines += ["", f"## Size and CPU latency ({bench['batch']}, {bench['torch_threads']} threads)", "",
                  "| | Params | File size | CPU latency |", "|---|---|---|---|",
                  f"| Teacher | {t['params'] / 1e6:.1f}M | {t['size_mb']:.0f} MB | {t['cpu_latency_ms']:.0f} ms |",
                  f"| Student | {s['params'] / 1e6:.2f}M | {s['size_mb']:.0f} MB | {s['cpu_latency_ms']:.0f} ms |",
                  f"| Ratio | {r['params']:.1f}× | {r['size_mb']:.1f}× | {r['cpu_latency_ms']:.1f}× |"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    root = Path(args.results)
    figs = root / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    runs = [load_json(p) for p in sorted((root / "runs").glob("*.json"))]
    teacher = load_json(root / "teacher.json")
    bench = load_json(root / "benchmark.json") if (root / "benchmark.json").exists() else None
    agg = aggregate(runs)

    save_json({"teacher": teacher, "benchmark": bench,
               "groups": [{"method": k[0], "num_train_stories": k[1], "temperature": k[2], "alpha": k[3], **v}
                          for k, v in sorted(agg.items(), key=lambda kv: str(kv[0]))]}, root / "summary.json")
    write_markdown(agg, teacher, bench, root / "RESULTS.md")
    if (root / "soft_labels.json").exists():
        fig_soft_labels(load_json(root / "soft_labels.json"), figs / "soft_labels.png")
    fig_training_curves(agg, teacher, figs / "training_curves.png")
    fig_data_scarcity(agg, teacher, figs / "data_scarcity.png")
    fig_temperature(agg, figs / "temperature.png")
    print(f"Wrote {root / 'RESULTS.md'}, {root / 'summary.json'} and {len(list(figs.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
