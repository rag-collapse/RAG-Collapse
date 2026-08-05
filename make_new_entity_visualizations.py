"""Regenerate entity collapse figures (unique entities + entity similarity) from the
consolidated `all_experiments/` data, written as `new_`-prefixed siblings next to the
existing figures in `visualization_outputs/` so old vs new can be compared side by side.

Only the 4 canonical models present in all_experiments are covered:
  Qwen2.5-14B-Instruct, DeepSeek-R1-Distill-Qwen-7B, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3

Source format = old aggregate entity-results JSON:
  aggregate_statistics.mean_unique_entities_per_round / mean_entity_similarity_per_round
(the exact two series the existing entity PNGs plot).
"""
import json
import os

import matplotlib.pyplot as plt

AE = "all_experiments/graphite"          # entity data only exists for the graphite dataset
VIZ = "visualization_outputs"

# Consistent colors (match the notebooks' COLORS dict; paraphrase variants reuse
# their baseline-regime color so the rerun-paraphrase figure lines up with baseline).
COLORS = {
    "Replace All":            "#1f77b4",
    "Replace One":            "#ff7f0e",
    "Search":                 "#2ca02c",
    "Rerank":                 "#d62728",
    "Search Paraphrase":      "#C2185B",
    "Agentic RAG":            "#9467bd",
    "Replace All (Para)":     "#1f77b4",
    "Replace One (Para)":     "#ff7f0e",
    "Search (Para)":          "#2ca02c",
    "Replace All + Paraphrase": "#d62728",
}

# org/model strings and the reranker's joined-form folder name.
MODELS = {
    "Qwen":     ("Qwen/Qwen2.5-14B-Instruct",                 "Qwen__Qwen2.5-14B-Instruct"),
    "DeepSeek": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",   "deepseek-ai__DeepSeek-R1-Distill-Qwen-7B"),
    "Llama":    ("meta-llama/Llama-3.1-8B-Instruct",          None),
    "Mistral":  ("mistralai/Mistral-7B-Instruct-v0.3",        None),
}


# ── all_experiments path builders ──
def p_baseline(om, regime, subdir=None):
    sub = f"{subdir}/" if subdir else ""
    return f"{AE}/baseline/{regime}/entity_extraction_output/{om}/{sub}local_{regime}_entity_results.json"

def p_agentic(om):
    return f"{AE}/agentic_rag/entity_extraction_output/{om}/local_agentic_rag_entity_results.json"

def p_paraphrase(om, pv):
    return f"{AE}/paraphrase/{pv}/entity_extraction_output/{om}/rerun-paraphrase/cpu_{pv}_paraphrased_entity_results.json"

def p_rerank(joined):
    if joined is None:
        return None
    return f"{AE}/rerank/entity_extraction_output/{joined}/rerank_lambda0.7_entity_results.json"


def build_figures():
    """Return a list of figure specs: {outdir, series:[(label,path)]}."""
    figs = []
    for name, (om, joined) in MODELS.items():
        # A) Top-level MAIN comparison — all available variants
        figs.append({
            "outdir": f"{VIZ}/{om}",
            "what": f"{name} — main comparison (all available variants)",
            "series": [
                ("Replace All",       p_baseline(om, "replace_all")),
                ("Replace One",       p_baseline(om, "replace_one")),
                ("Search",            p_baseline(om, "search")),
                ("Agentic RAG",       p_agentic(om)),
                ("Rerank",            p_rerank(joined)),
                ("Search Paraphrase", p_paraphrase(om, "search")),
            ],
        })
        # F) rerun-paraphrase — the three paraphrase-mitigation variants (all 4 models)
        figs.append({
            "outdir": f"{VIZ}/{om}/rerun-paraphrase",
            "what": f"{name} — paraphrase mitigation variants",
            "series": [
                ("Replace All (Para)", p_paraphrase(om, "hybrid")),
                ("Replace One (Para)", p_paraphrase(om, "replace_one")),
                ("Search (Para)",      p_paraphrase(om, "search")),
            ],
        })

    Qom, Qjoined = MODELS["Qwen"][0], MODELS["Qwen"][1]
    Dom = MODELS["DeepSeek"][0]
    Lom = MODELS["Llama"][0]
    Mom = MODELS["Mistral"][0]

    # B) baseline/ subfolder (DeepSeek, Llama) — baseline 3 variants
    for om, label in [(Dom, "DeepSeek"), (Lom, "Llama")]:
        figs.append({
            "outdir": f"{VIZ}/{om}/baseline",
            "what": f"{label} — baseline variants",
            "series": [
                ("Replace All", p_baseline(om, "replace_all")),
                ("Replace One", p_baseline(om, "replace_one")),
                ("Search",      p_baseline(om, "search")),
            ],
        })

    # C) comparison/ (Qwen) — baseline 3 + Agentic RAG
    figs.append({
        "outdir": f"{VIZ}/{Qom}/comparison",
        "what": "Qwen — baseline + Agentic RAG",
        "series": [
            ("Replace All", p_baseline(Qom, "replace_all")),
            ("Replace One", p_baseline(Qom, "replace_one")),
            ("Search",      p_baseline(Qom, "search")),
            ("Agentic RAG", p_agentic(Qom)),
        ],
    })

    # D) agentic_rag/ (Qwen) — Agentic RAG alone
    figs.append({
        "outdir": f"{VIZ}/{Qom}/agentic_rag",
        "what": "Qwen — Agentic RAG only",
        "series": [("Agentic RAG", p_agentic(Qom))],
    })

    # E) older_experiments/ (Qwen) — baseline 3 from the older_experiments subdir
    figs.append({
        "outdir": f"{VIZ}/{Qom}/older_experiments",
        "what": "Qwen — baseline variants (older_experiments)",
        "series": [
            ("Replace All", p_baseline(Qom, "replace_all", "older_experiments")),
            ("Replace One", p_baseline(Qom, "replace_one", "older_experiments")),
            ("Search",      p_baseline(Qom, "search",      "older_experiments")),
        ],
    })

    # G) search_para_compare/ (Mistral) — Search baseline vs Search paraphrase
    figs.append({
        "outdir": f"{VIZ}/{Mom}/search_para_compare",
        "what": "Mistral — Search vs Search Paraphrase",
        "series": [
            ("Search",            p_baseline(Mom, "search")),
            ("Search Paraphrase", p_paraphrase(Mom, "search")),
        ],
    })

    # H) mitigation-paraphrase-all-references/ (Qwen, Mistral)
    #    = Replace All baseline vs Replace All with paraphrasing of all references on.
    figs.append({
        "outdir": f"{VIZ}/mitigation-paraphrase-all-references/{Qom}",
        "what": "Qwen — Replace All: baseline vs paraphrase-all-references",
        "series": [
            ("Replace All",             p_baseline(Qom, "replace_all")),
            ("Replace All + Paraphrase", p_baseline(Qom, "replace_all", "paraphrase_on")),
        ],
    })
    figs.append({
        "outdir": f"{VIZ}/mitigation-paraphrase-all-references/{Mom}",
        "what": "Mistral — Replace All: baseline vs paraphrase-all-references",
        "series": [
            ("Replace All",             p_baseline(Mom, "replace_all")),
            ("Replace All + Paraphrase", p_baseline(Mom, "replace_all", "paraphrase-on")),
        ],
    })
    return figs


METRICS = [
    ("mean_unique_entities_per_round", "Unique Entities",  "new_unique_entities_per_round.png"),
    ("mean_entity_similarity_per_round", "Entity Similarity", "new_entity_similarity_per_round.png"),
]


def plot_figure(spec):
    # Load only existing series.
    loaded = []
    skipped = []
    for label, path in spec["series"]:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                loaded.append((label, json.load(f)["aggregate_statistics"]))
        else:
            skipped.append(label)

    if not loaded:
        print(f"  [SKIP figure] {spec['what']}: no source files found")
        return 0

    os.makedirs(spec["outdir"], exist_ok=True)
    n_written = 0
    for metric_key, ylabel, fname in METRICS:
        fig, ax = plt.subplots(figsize=(10, 6))
        for label, agg in loaded:
            vals = agg[metric_key]
            ax.plot(range(1, len(vals) + 1), vals, linewidth=2.5,
                    label=label, color=COLORS.get(label))
        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{ylabel} Per Round", fontsize=14)
        ax.legend()
        plt.tight_layout()
        out = f"{spec['outdir']}/{fname}"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        n_written += 1

    inc = ", ".join(lbl for lbl, _ in loaded)
    msg = f"  [OK] {spec['what']}\n       -> {spec['outdir']}/  ({inc})"
    if skipped:
        msg += f"\n       (no data, omitted: {', '.join(skipped)})"
    print(msg)
    return n_written


def main():
    figs = build_figures()
    total = 0
    print(f"Generating new_ entity figures for {len(figs)} figure specs...\n")
    for spec in figs:
        total += plot_figure(spec)
    print(f"\nDone. Wrote {total} PNG files.")


if __name__ == "__main__":
    main()
