"""R4 (reviewer NbXB): re-plot collapse against contamination fraction instead of round.

The regimes reach a given contamination level (fraction of context that is AI-generated) at
very different rounds -- Replace-All saturates in one round, Replace-One ramps ~linearly,
Search ramps gradually. Plotting collapse vs *round* therefore compares regimes at unequal
saturation. This re-plots the already-logged metrics with contamination fraction on the x-axis
so the three regimes are compared at equivalent saturation.

Pure re-plot. No recomputation:
  x  = mean_q ai_reference_percentage        (contamination fraction, evaluation_outputs)
  y1 = mean_q unique_entities                (entity diversity; falls with collapse; entity_extraction_output)
  y2 = mean_q same_answer_percentage         (answer convergence; rises with collapse; evaluation_outputs)

Entity *similarity* is intentionally NOT used as the collapse axis: this repo scores an
empty-entity answer 0.0 ("diverse"), so degenerate answers pull similarity *down*, making it a
confounded collapse signal (see docs/entity_extraction_comparison.md). Unique-entity count and
same-answer% are monotone in collapse and unambiguous.
"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
BASE = os.path.join(ROOT, "all_experiments", "graphite", "baseline")
OUT = os.path.join(ROOT, "camera_ready_outputs", "R4")
os.makedirs(OUT, exist_ok=True)

MODELS = [
    ("Qwen/Qwen2.5-14B-Instruct", "Qwen2.5-14B"),
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7B"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1-7B"),
]
REGIMES = [("replace_all", "Replace-All"), ("replace_one", "Replace-One"), ("search", "Search")]
COLORS = {"Replace-All": "#d62728", "Replace-One": "#1f77b4", "Search": "#2ca02c"}


def find_eval(regime, model):
    # Mistral evals live under a para_off/ (baseline) or paraphrase_on/ subdir; prefer para_off.
    direct = os.path.join(BASE, regime, "evaluation_outputs", model, f"local_{regime}_eval.json")
    if os.path.exists(direct):
        return direct
    for sub in ("para_off", "*"):
        c = sorted(glob.glob(os.path.join(BASE, regime, "evaluation_outputs", model, sub, f"local_{regime}_eval.json")))
        c = [p for p in c if "paraphrase_on" not in p]
        if c:
            return c[0]
    return None


def mean_by_round(questions, getter):
    """getter(iteration_dict) -> float or None; returns list indexed by iteration_number."""
    acc = {}
    for q in questions:
        for it in q["iterations"]:
            r = it["iteration_number"]
            v = getter(it)
            if v is not None:
                acc.setdefault(r, []).append(v)
    rounds = sorted(acc)
    return rounds, [float(np.mean(acc[r])) for r in rounds]


def load_series(regime, model):
    ev = find_eval(regime, model)
    en = os.path.join(BASE, regime, "entity_extraction_output", model, f"local_{regime}_entity_results.json")
    if not ev or not os.path.exists(en):
        return None
    evd = json.load(open(ev)); end = json.load(open(en))
    r_c, contam = mean_by_round(evd["questions"], lambda it: it["metrics"].get("ai_reference_percentage"))
    _,   same   = mean_by_round(evd["questions"], lambda it: it["metrics"].get("same_answer_percentage"))
    r_u, uniq   = mean_by_round(end["questions"], lambda it: it.get("unique_entities"))
    # align on common rounds
    cmap = dict(zip(r_c, contam)); smap = dict(zip(r_c, same)); umap = dict(zip(r_u, uniq))
    rounds = sorted(set(r_c) & set(r_u))
    return {
        "rounds": rounds,
        "contam": [cmap[r] for r in rounds],
        "same":   [smap[r] for r in rounds],
        "uniq":   [umap[r] for r in rounds],
    }


def panel(ax, model_token, ykey, ylabel):
    got = False
    for regime, rlabel in REGIMES:
        s = load_series(regime, model_token)
        if not s:
            continue
        got = True
        ax.plot(s["contam"], s[ykey], "-o", ms=3, lw=1.5, color=COLORS[rlabel], label=rlabel)
    ax.set_xlabel("contamination fraction (AI docs in context)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return got


for ykey, ylabel, fname in [
    ("uniq", "mean unique entities / round", "unique_entities_vs_contamination"),
    ("same", "same-answer % (answer convergence)", "same_answer_vs_contamination"),
]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (mt, mn) in zip(axes.ravel(), MODELS):
        panel(ax, mt, ykey, ylabel)
        ax.set_title(mn)
    axes.ravel()[0].legend(fontsize=9)
    fig.suptitle(f"R4: collapse vs contamination fraction ({ylabel})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(OUT, fname + ".png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print("wrote", p)

    # Qwen-only primary panel
    fig, ax = plt.subplots(figsize=(6.5, 5))
    panel(ax, "Qwen/Qwen2.5-14B-Instruct", ykey, ylabel)
    ax.set_title(f"Qwen2.5-14B — {ylabel} vs contamination")
    ax.legend()
    p = os.path.join(OUT, fname + "_qwen.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print("wrote", p)

# also dump the aligned series as JSON for the write-up / reproducibility
dump = {}
for mt, mn in MODELS:
    dump[mn] = {}
    for regime, rlabel in REGIMES:
        s = load_series(regime, mt)
        if s:
            dump[mn][rlabel] = s
json.dump(dump, open(os.path.join(OUT, "r4_series.json"), "w"), indent=1)
print("wrote", os.path.join(OUT, "r4_series.json"))
print("### R4 done ###")
