#!/usr/bin/env python3
"""Figures for the 1400q baseline-match HotpotQA distractor runs, for docs/hotpotqa_diverse_synth.html.

Reads the compare_sweep summaries (fetched locally to SUMDIR as <mode>/<model>/sweep_summary_<variant>.json)
and writes PNGs to FIGDIR. Full per-variant rounds (search=30 / replace_one=20 / replace_all=10) — unlike
the earlier 400q/3-round figures.

Produces three groups:
  (A) REGEN  — Qwen2.5-14B, diverse_synth vs equal_diverse_synth: per-round dynamics (metric x round)
               and dose-response at the final round (metric x fraction/topic). Mirrors the old 12 figures.
  (B) DOSE   — all 4 models, final-round metric x fraction/topic, faceted by variant (one line per model).
  (C) TRAJ   — all 4 models, metric x round at the strongest arm, faceted by variant (one line per model).
"""
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUMDIR = pathlib.Path("1400q_summaries")                       # fetched: <mode>/<model>/sweep_summary_<variant>.json
FIGDIR = pathlib.Path("visualization_outputs/hotpot_1400q_figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

MODELS = ["qwen2.5-14b", "llama-3.1-8b", "mistral-7b", "deepseek-r1-distill-qwen-7b"]
MLABEL = {"qwen2.5-14b": "Qwen2.5-14B", "llama-3.1-8b": "Llama-3.1-8B",
          "mistral-7b": "Mistral-7B", "deepseek-r1-distill-qwen-7b": "DeepSeek-R1-7B"}
MCOL = {"qwen2.5-14b": "#2563eb", "llama-3.1-8b": "#0d9488",
        "mistral-7b": "#d97706", "deepseek-r1-distill-qwen-7b": "#be185d"}
VARIANTS = ["search", "replace_one", "replace_all"]
VLABEL = {"search": "search (growing store)", "replace_one": "replace_one", "replace_all": "replace_all"}
DFRACS = ["0", "0.3", "0.5", "0.7"]
ETOPICS = ["0", "1", "2", "3", "4"]
DCOL = {"0": "#000000", "0.3": "#2563eb", "0.5": "#d97706", "0.7": "#7c3aed"}
ECOL = {"0": "#000000", "1": "#0d9488", "2": "#2563eb", "3": "#d97706", "4": "#be185d"}
VC = {"search": "#7c3aed", "replace_one": "#b45309", "replace_all": "#0d9488"}
REF = "#dc2626"
METRICS = [
    ("gold_match", "gold match (accuracy)", False, "Gold match"),
    ("distractor_adoption", "distractor adoption", False, "Distractor adoption"),
    ("distinct_answers", "distinct answers / question", True, "Distinct answers"),
]
MSHORT = {"gold_match": "gold", "distractor_adoption": "adopt", "distinct_answers": "distinct"}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "savefig.bbox": "tight", "savefig.dpi": 120, "figure.dpi": 120,
    "axes.edgecolor": "#c6ccd8", "grid.color": "#e4e7ee", "axes.grid": True, "grid.alpha": 1.0,
    "font.size": 11, "axes.titlesize": 12.5, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2.2, "lines.markersize": 5,
})


def load(mode, model, var):
    p = SUMDIR / mode / model / f"sweep_summary_{var}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get("fractions", {})


def rounds_of(arm_dict):
    """Sorted int rounds present in an arm's per-round dict."""
    return sorted(int(r) for r in arm_dict)


def final_round(fr, arm):
    """Highest round key available for a given arm (as str), or None."""
    a = fr.get(arm)
    if not a:
        return None
    return str(max(int(r) for r in a))


def _topic_label(t):
    return "0 (baseline)" if t == "0" else f"{t} topic" + ("" if t == "1" else "s")


# ---------------- (A) REGEN: Qwen diverse vs equal ----------------
def regen_qwen():
    div = {v: load("diverse_synth", "qwen2.5-14b", v) for v in VARIANTS}
    eq = {v: load("equal_diverse_synth", "qwen2.5-14b", v) for v in VARIANTS}
    n = 0
    # per-round dynamics (full rounds)
    for metric, ylab, ref1, title in METRICS:
        for v in VARIANTS:
            if not div.get(v) or not eq.get(v):
                continue
            fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, layout="constrained")
            for f in DFRACS:
                a = div[v].get(f)
                if not a:
                    continue
                rs = rounds_of(a)
                axL.plot(rs, [a[str(r)][metric] for r in rs], marker="o", color=DCOL[f], label=f"f = {f}")
            for t in ETOPICS:
                a = eq[v].get(t)
                if not a:
                    continue
                rs = rounds_of(a)
                axR.plot(rs, [a[str(r)][metric] for r in rs], marker="o", color=ECOL[t], label=_topic_label(t))
            for ax, ttl in ((axL, "diverse_synth (fraction sweep)"), (axR, "equal_diverse_synth (topic sweep)")):
                ax.set_title(ttl)
                ax.set_xlabel("round")
                if ref1:
                    ax.axhline(1.0, color=REF, ls=":", lw=1.2)
                ax.legend(fontsize=8, ncol=2)
            axL.set_ylabel(ylab)
            fig.suptitle(f"{title} across rounds — Qwen2.5-14B, {VLABEL[v]}", fontsize=14)
            fig.savefig(FIGDIR / f"qwen_cmp_{MSHORT[metric]}_{v}.png")
            plt.close(fig)
            n += 1
    # dose-response @ final round
    for metric, ylab, ref1, title in METRICS:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, layout="constrained")
        for v in VARIANTS:
            if div.get(v):
                xs = [float(f) for f in DFRACS if div[v].get(f)]
                ys = [div[v][f][final_round(div[v], f)][metric] for f in DFRACS if div[v].get(f)]
                axL.plot(xs, ys, marker="o", color=VC[v], label=VLABEL[v])
            if eq.get(v):
                xs = [int(t) for t in ETOPICS if eq[v].get(t)]
                ys = [eq[v][t][final_round(eq[v], t)][metric] for t in ETOPICS if eq[v].get(t)]
                axR.plot(xs, ys, marker="o", color=VC[v], label=VLABEL[v])
        axL.set_title("diverse_synth"); axL.set_xlabel("distractor fraction")
        axR.set_title("equal_diverse_synth"); axR.set_xlabel("number of topics")
        for ax in (axL, axR):
            if ref1:
                ax.axhline(1.0, color=REF, ls=":", lw=1.2)
            ax.legend(fontsize=8)
        axL.set_ylabel(ylab)
        fig.suptitle(f"{title} @ final round — Qwen2.5-14B", fontsize=14)
        fig.savefig(FIGDIR / f"qwen_dose_{MSHORT[metric]}.png")
        plt.close(fig)
        n += 1
    return n


# ---------------- (B) DOSE: all models, final round, faceted by variant ----------------
def dose_all_models():
    n = 0
    for mode, arms, xlab in (("diverse_synth", DFRACS, "distractor fraction"),
                             ("equal_diverse_synth", ETOPICS, "number of topics")):
        for metric, ylab, ref1, title in METRICS:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True, layout="constrained")
            for ax, v in zip(axes, VARIANTS):
                for model in MODELS:
                    fr = load(mode, model, v)
                    if not fr:
                        continue
                    xs, ys = [], []
                    for arm in arms:
                        frnd = final_round(fr, arm)
                        if frnd is None:
                            continue
                        xs.append(float(arm) if mode == "diverse_synth" else int(arm))
                        ys.append(fr[arm][frnd][metric])
                    if xs:
                        ax.plot(xs, ys, marker="o", color=MCOL[model], label=MLABEL[model])
                ax.set_title(VLABEL[v]); ax.set_xlabel(xlab)
                if ref1:
                    ax.axhline(1.0, color=REF, ls=":", lw=1.2)
            axes[0].set_ylabel(ylab)
            axes[-1].legend(fontsize=8)
            fig.suptitle(f"{title} @ final round vs {xlab} — all models ({mode})", fontsize=14)
            fig.savefig(FIGDIR / f"dose_{mode}_{MSHORT[metric]}.png")
            plt.close(fig)
            n += 1
    return n


# ---------------- (C) TRAJ: all models, strongest arm, metric x round, faceted by variant ----------------
def traj_all_models():
    n = 0
    for mode, strong in (("diverse_synth", "0.7"), ("equal_diverse_synth", "4")):
        for metric, ylab, ref1, title in METRICS:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True, layout="constrained")
            for ax, v in zip(axes, VARIANTS):
                for model in MODELS:
                    fr = load(mode, model, v)
                    if not fr or not fr.get(strong):
                        continue
                    a = fr[strong]
                    rs = rounds_of(a)
                    ax.plot(rs, [a[str(r)][metric] for r in rs], color=MCOL[model], label=MLABEL[model])
                ax.set_title(VLABEL[v]); ax.set_xlabel("round")
                if ref1:
                    ax.axhline(1.0, color=REF, ls=":", lw=1.2)
            axes[0].set_ylabel(ylab)
            axes[-1].legend(fontsize=8)
            arm_desc = f"fraction {strong}" if mode == "diverse_synth" else f"{strong} topics"
            fig.suptitle(f"{title} across rounds ({arm_desc}) — all models ({mode})", fontsize=14)
            fig.savefig(FIGDIR / f"traj_{mode}_{MSHORT[metric]}.png")
            plt.close(fig)
            n += 1
    return n


if __name__ == "__main__":
    a = regen_qwen()
    b = dose_all_models()
    c = traj_all_models()
    total = len(list(FIGDIR.glob("*.png")))
    print(f"REGEN(qwen)={a}  DOSE(all)={b}  TRAJ(all)={c}  ->  {total} PNGs in {FIGDIR}")
