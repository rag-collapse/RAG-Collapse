#!/usr/bin/env python3
"""Side-by-side comparison figures: diverse_synth (fraction sweep) vs equal_diverse_synth
(topic sweep), per variant and metric, from the two local (gitignored) result folders.

  distractor_sweep/        diverse_synth   arms = fractions {0,0.3,0.5,0.7}
  equal_distractor_sweep/  equal_synth     arms = #topics   {0,1,2,3,4}

Writes 12 PNGs (light theme, black f=0/t=0 baseline) to
visualization_outputs/equal_vs_diverse_figures/ — embedded into docs/hotpotqa_diverse_synth.html.
"""
import json
import pathlib
import matplotlib.pyplot as plt

DIV = pathlib.Path("distractor_sweep")
EQ = pathlib.Path("equal_distractor_sweep")
FIGDIR = pathlib.Path("visualization_outputs/equal_vs_diverse_figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

VARIANTS = ["search", "hybrid", "replace_one"]
VLABEL = {"search": "search (growing store)", "hybrid": "hybrid / replace-all", "replace_one": "replace_one"}
ROUNDS = [0, 1, 2]
DFRACS = ["0", "0.3", "0.5", "0.7"]
ETOPICS = ["0", "1", "2", "3", "4"]

INK, MUTE, GRID, EDGE = "#1f2430", "#5b6472", "#e4e7ee", "#c6ccd8"
DCOL = {"0": "#000000", "0.3": "#2563eb", "0.5": "#d97706", "0.7": "#7c3aed"}      # diverse: black + blue/amber/violet
ECOL = {"0": "#000000", "1": "#0d9488", "2": "#2563eb", "3": "#d97706", "4": "#be185d"}  # equal: black + teal/blue/amber/magenta
VC = {"search": "#7c3aed", "hybrid": "#0d9488", "replace_one": "#b45309"}
REF = "#dc2626"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "savefig.bbox": "tight", "savefig.dpi": 120, "figure.dpi": 120,
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.edgecolor": EDGE, "xtick.color": MUTE, "ytick.color": MUTE,
    "grid.color": GRID, "axes.grid": True, "grid.alpha": 1.0, "font.size": 11,
    "axes.titlesize": 12.5, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2.2, "lines.markersize": 6,
})

div = {v: json.loads((DIV / f"sweep_summary_{v}.json").read_text())["fractions"] for v in VARIANTS}
eq = {v: json.loads((EQ / f"sweep_summary_{v}.json").read_text())["fractions"] for v in VARIANTS}


def ser(summ, v, arm, m):
    return [summ[v][arm][str(r)][m] for r in ROUNDS]


METRICS = [  # (key, y-label, ymax, reference-line-at-1.0, title)
    ("gold_match", "gold match (accuracy)", 0.6, False, "Gold match"),
    ("distractor_adoption", "distractor adoption", 0.85, False, "Distractor adoption"),
    ("distinct_answers", "distinct answers / question", 2.0, True, "Distinct answers"),
]
MSHORT = {"gold_match": "gold", "distractor_adoption": "adopt", "distinct_answers": "distinct"}


def _topic_label(t):
    return "0 (baseline)" if t == "0" else f"{t} topic" + ("" if t == "1" else "s")


# --- per-round dynamics: diverse | equal, side by side, one figure per (metric, variant) ---
for metric, ylab, ymax, ref1, title in METRICS:
    for v in VARIANTS:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, layout="constrained")
        for f in DFRACS:
            axL.plot(ROUNDS, ser(div, v, f, metric), marker="o", color=DCOL[f], label=f"f = {f}")
        for t in ETOPICS:
            axR.plot(ROUNDS, ser(eq, v, t, metric), marker="o", color=ECOL[t], label=_topic_label(t))
        for ax, ttl in ((axL, "diverse_synth (fraction sweep)"), (axR, "equal_diverse_synth (topic sweep)")):
            ax.set_title(ttl)
            ax.set_xlabel("round")
            ax.set_xticks(ROUNDS)
            if ref1:
                ax.axhline(1.0, color=REF, ls=":", lw=1.2)
            ax.legend(fontsize=8, ncol=2)
        axL.set_ylabel(ylab)
        axL.set_ylim(0.8 if ref1 else -0.02, ymax)
        fig.suptitle(f"{title} across rounds — {VLABEL[v]}", fontsize=14)
        fig.savefig(FIGDIR / f"cmp_{MSHORT[metric]}_{v}.png")
        plt.close(fig)

# --- dose-response @ final round (r2): diverse-vs-fraction | equal-vs-#topics, per metric ---
dxf = [float(f) for f in DFRACS]
ext = [int(t) for t in ETOPICS]
for metric, ylab, ymax, ref1, title in METRICS:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, layout="constrained")
    for v in VARIANTS:
        axL.plot(dxf, [div[v][f]["2"][metric] for f in DFRACS], marker="o", color=VC[v], label=VLABEL[v])
        axR.plot(ext, [eq[v][t]["2"][metric] for t in ETOPICS], marker="o", color=VC[v], label=VLABEL[v])
    axL.set_title("diverse_synth")
    axL.set_xlabel("distractor fraction")
    axL.set_xticks(dxf)
    axR.set_title("equal_diverse_synth")
    axR.set_xlabel("number of topics")
    axR.set_xticks(ext)
    for ax in (axL, axR):
        if ref1:
            ax.axhline(1.0, color=REF, ls=":", lw=1.2)
        ax.legend(fontsize=8)
    axL.set_ylabel(ylab)
    axL.set_ylim(0.8 if ref1 else -0.02, ymax)
    fig.suptitle(f"{title} @ final round (r2)", fontsize=14)
    fig.savefig(FIGDIR / f"cmp_dose_{MSHORT[metric]}.png")
    plt.close(fig)

print("wrote", len(list(FIGDIR.glob("*.png"))), "figures to", FIGDIR)
