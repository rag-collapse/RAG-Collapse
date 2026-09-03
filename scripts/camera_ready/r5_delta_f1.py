"""R5 (all reviewers): downstream harm as per-model effect sizes, not a binary.

For each model x regime on HotpotQA, compute per-question F1 at round 0 (clean, human-only
context) and at the final round, then Delta F1 = F1_final - F1_0. Report the mean Delta F1 with a
95% bootstrap CI across the 1400 questions, so "robust only for Qwen2.5-14B" becomes the more
precise "directionally consistent, significant in k of 4, underpowered."

Data already logged: evaluation_outputs/.../*_hotpot_eval.json -> questions[].iterations[].metrics.avg_f1
(avg over the round's runs, per question). Pure re-analysis.

Power note: HotpotQA gold answers are very short (often 1-3 tokens), so token-F1 is coarse and
per-round movements are small relative to question-level variance -- CIs are wide by construction,
which is the point: the effect is real and directionally consistent but the test is underpowered.
"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
BASE = os.path.join(ROOT, "all_experiments", "hotpotqa", "baseline")
OUT = os.path.join(ROOT, "camera_ready_outputs", "R5")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)

MODELS = [
    ("Qwen/Qwen2.5-14B-Instruct", "Qwen2.5-14B"),
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7B"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1-7B"),
]
REGIMES = [("search", "Search"), ("replace_one", "Replace-One"), ("replace_all", "Replace-All")]


def find_detailed_eval(regime, model):
    cands = sorted(glob.glob(os.path.join(BASE, regime, "evaluation_outputs", model, "*hotpot_eval.json")))
    for p in cands:
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if isinstance(d, dict) and "questions" in d and d["questions"] and "iterations" in d["questions"][0]:
            return p, d
    return None, None


def per_question_delta(d):
    """Return arrays (f1_0, f1_final) over questions that have both round 0 and a final round."""
    f0, ff = [], []
    for q in d["questions"]:
        its = {it["iteration_number"]: it["metrics"].get("avg_f1") for it in q["iterations"]}
        if not its:
            continue
        r0 = min(its); rf = max(its)
        if its.get(r0) is None or its.get(rf) is None or rf == r0:
            continue
        f0.append(its[r0]); ff.append(its[rf])
    return np.array(f0), np.array(ff)


def boot_ci(delta, n=10000):
    idx = rng.integers(0, len(delta), size=(n, len(delta)))
    means = delta[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


rows = []
for regime, rlabel in REGIMES:
    for mt, mn in MODELS:
        p, d = find_detailed_eval(regime, mt)
        if not d:
            rows.append((rlabel, mn, None)); continue
        f0, ff = per_question_delta(d)
        delta = ff - f0
        mean = float(delta.mean())
        lo, hi = boot_ci(delta)
        rf = max(it["iteration_number"] for it in d["questions"][0]["iterations"])
        sig = "yes" if (lo > 0 or hi < 0) else "no"
        rows.append((rlabel, mn, dict(n=len(delta), f0=float(f0.mean()), ff=float(ff.mean()),
                                      mean=mean, lo=float(lo), hi=float(hi), rf=rf, sig=sig)))

# ---- text table ----
print(f"{'regime':12s} {'model':14s} {'n':>5s} {'F1_0':>7s} {'F1_fin':>7s} {'dF1':>8s}  {'95% CI':>18s}  sig  rounds")
for rlabel, mn, r in rows:
    if r is None:
        print(f"{rlabel:12s} {mn:14s}   (no detailed eval found)"); continue
    print(f"{rlabel:12s} {mn:14s} {r['n']:5d} {r['f0']:7.3f} {r['ff']:7.3f} {r['mean']:+8.3f}  [{r['lo']:+.3f}, {r['hi']:+.3f}]  {r['sig']:>3s}  0..{r['rf']}")

# ---- forest plot (one row per model, grouped by regime) ----
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True)
for ax, (regime, rlabel) in zip(axes, REGIMES):
    ys = []
    for i, (mt, mn) in enumerate(MODELS):
        r = next((x[2] for x in rows if x[0] == rlabel and x[1] == mn), None)
        if not r:
            continue
        y = len(MODELS) - i
        ys.append((y, mn))
        color = "#1f77b4" if r["sig"] == "yes" else "#999999"
        ax.plot([r["lo"], r["hi"]], [y, y], "-", color=color, lw=2)
        ax.plot(r["mean"], y, "o", color=color, ms=6)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_yticks([y for y, _ in ys]); ax.set_yticklabels([m for _, m in ys])
    ax.set_title(rlabel); ax.set_xlabel("Δ F1 (final − round 0)"); ax.grid(alpha=0.3, axis="x")
fig.suptitle("R5: downstream ΔF1 by model (95% bootstrap CI over 1400 questions)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
p = os.path.join(OUT, "delta_f1_forest.png")
fig.savefig(p, dpi=150, bbox_inches="tight")
print("\nwrote", p)

json.dump([(rl, mn, r) for rl, mn, r in rows], open(os.path.join(OUT, "r5_delta_f1.json"), "w"), indent=1)
print("### R5 done ###")
