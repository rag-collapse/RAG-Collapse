# GEPA Prompt Optimization: Architecture and Metrics

## High-Level Loop

```mermaid
flowchart TD
    A([Start: seed_candidate\nGEPA_RAG_GENERATION_SYSTEM_PROMPT]) --> B

    B[gepa.optimize\nmax_metric_calls budget\npareto selection strategy]

    B --> C[evaluate on trainset\n60 questions]
    C --> D[make_reflective_dataset\nbuild structured failure report]
    D --> E[reflection_lm\nclaude-opus-4-1\nproposes new system_prompt]
    E --> F[evaluate on valset\n20 questions]
    F --> G{Pareto check\nnew candidate vs frontier\non anti_collapse × quality}
    G -->|not dominated| H[Add to Pareto frontier]
    G -->|dominated| I[Discard]
    H --> J{budget exhausted?\nmax_metric_calls}
    I --> J
    J -->|No| C
    J -->|Yes| K([result.best_candidate\napply_best_prompt.py\nwrites back to formatters.py])
```

---

## evaluate(): Per Question

```mermaid
flowchart TD
    Q[RAGDataInst\nquestion + docs] --> OC[Build original_context\nget_context_str_from_docs\nchars_per_doc=800]

    OC --> SIM

    subgraph SIM [Run 3 Collapse Simulations, pipeline_simulator.py]
        RA[simulate_replace_all\nHybridConfig: 10 synth, 0 db\n10 rounds\nentire context replaced each round]
        RO[simulate_replace_one\nreal refs capped at 10\n10 rounds\none slot replaced per round\nslot = iteration mod len docs]
        SE[simulate_search\nChunkedRetrievalStore\nall-MiniLM-L6-v2 embeddings\n10 rounds\ntop-k=10 retrieved each round\nanswer added to store after each round]
    end

    SIM --> SC

    subgraph SC [Score Each Variant, scoring.py]
        AC[anti_collapse_score\nquestion + answers list\njudge_model via litellm]
        QJ[judge_quality_score\nquestion + original_context\nfinal answer only\njudge_model via litellm]
    end

    SC --> AVG[Average across variants\navg_anti_collapse\navg_quality]
    AVG --> EB[EvaluationBatch\nscores = avg_anti_collapse per question\nobjective_scores = anti_collapse × quality\ntrajectories = RAGTrace per question]
```

---

## Collapse Simulations: Round-by-Round

```mermaid
flowchart TD
    subgraph RA [replace_all, fastest collapse]
        RA0[Round 0: original refs as context] --> RAG0[LLM generates answer_0]
        RAG0 --> RA1[Round 1: answer_0 becomes ALL 10 docs]
        RA1 --> RAG1[LLM generates answer_1]
        RAG1 --> RA2[Round N: answer_{N-1} becomes ALL 10 docs\n...]
    end

    subgraph RO [replace_one, slow decay]
        RO0[Round 0: original refs as context up to 10] --> ROG0[LLM generates answer_0]
        ROG0 --> RO1[Round 1: slot 1 mod len replaced by answer_0]
        RO1 --> ROG1[LLM generates answer_1]
        ROG1 --> RO2[Round N: slot N mod len replaced by answer_{N-1}\noriginal refs decay out one slot at a time]
    end

    subgraph SE [search, retrieval contamination]
        SE0[Seed store with original refs] --> SEQ0[Retrieve top-10 by cosine similarity]
        SEQ0 --> SEG0[LLM generates answer_0]
        SEG0 --> SEA0[Add answer_0 to store]
        SEA0 --> SEQ1[Retrieve top-10, AI text now competes\nwith original refs for top slots]
        SEQ1 --> SEG1[LLM generates answer_1\n...]
    end
```

---

## anti_collapse_score: Step by Step

```mermaid
flowchart TD
    IN[Input: question + answers list\none answer per simulation round] --> CLEAN[Filter empty answers\nrequires ≥ 2 non-empty]

    CLEAN --> EXT

    subgraph EXT [Step 1, Entity Extraction]
        EXT1[One litellm call per answer\nENTITY_EXTRACTION_SYSTEM_PROMPT\nENTITY_EXTRACTION_USER_PROMPT\ntemperature=0, max_tokens=256]
        EXT1 --> EXT2[Returns list of entity strings per answer\nper_answer_entities: list of list of str]
    end

    EXT --> UNION[Collect all unique mentions\nacross all answers\nall_mentions = set union]

    UNION --> CLUSTER

    subgraph CLUSTER [Step 2, Entity Clustering]
        CL1[One litellm call for all mentions\nENTITY_CLUSTERING_SYSTEM_PROMPT\nENTITY_CLUSTERING_USER_PROMPT\ntemperature=0, max_tokens=512]
        CL1 --> CL2[Returns canonical_map\ncanonical_name → list of surface mentions\nfallback: each mention maps to itself]
    end

    CLUSTER --> VEC

    subgraph VEC [Step 3, Binary Entity Vectors]
        V1[Build mention_to_canonical lookup\nlowercased keys]
        V1 --> V2[For each answer build binary vector\nlength = number of canonical entities\nvec_i = 1.0 if canonical_i mentioned else 0.0]
    end

    VEC --> SIM

    subgraph SIM [Step 4, Pairwise Cosine Similarity]
        S1[compute_entity_similarity from entity_extraction.py\nall pairs of entity vectors\ncosine similarity for each pair]
        S1 --> S2[Returns mean_entity_similarity\nalso min, max, std, not used here]
    end

    SIM --> SCORE[Step 5, Invert\nscore = 1.0 − mean_entity_similarity\nunique_entity_count = len canonical_map]

    SCORE --> OUT([Returns: score float, unique_entity_count int\nHigher score = more diverse entity sets = less collapse\nLower score = same entities repeated = collapsed])
```

---

## judge_quality_score: Step by Step

```mermaid
flowchart TD
    IN2[Input: question\noriginal_context from round 0 docs\nfinal_answer from last simulation round] --> CALL[One litellm call\ntemperature=0, max_tokens=128\nReturns JSON]

    CALL --> PARSE[Parse JSON\nfaithfulness: 0.0–1.0\nrelevance: 0.0–1.0]

    PARSE --> WEIGHT[Weighted sum\n0.6 × faithfulness\n+ 0.4 × relevance]

    WEIGHT --> OUT2([Returns: quality float 0.0–1.0\nFaithfulness weighted higher because\ngrounding is the primary concern\nFallback: 0.5 on any error])

    note1[Faithfulness: are all claims supported\nby the ORIGINAL uncontaminated context?\n1.0 = fully grounded\n0.0 = hallucinated or drifted] -.-> PARSE
    note2[Relevance: does the answer address\nthe question directly?\n1.0 = complete and focused\n0.0 = off-topic] -.-> PARSE
```

---

## make_reflective_dataset: Feedback to Reflection LM

```mermaid
flowchart TD
    TR[RAGTrace per question\noriginal_context\nresults per variant\nVariantResult: answers, anti_collapse, unique_entities, quality] --> CHECK

    subgraph CHECK [Threshold Checks]
        C1{anti_collapse < 0.30?}
        C2{quality < 0.50?}
    end

    C1 -->|Yes| D1[Diagnosis: answer lacks entity diversity,\nthe prompt may cause the model to fixate\non a narrow subset of entities]
    C1 -->|No| D3[Scores acceptable]
    C2 -->|Yes| D2[Diagnosis: Final answer drifted\nfrom original context, the prompt\nmay not anchor the model to retrieved facts]

    D1 --> REC[Build record per question\nInputs: System Prompt only\nGenerated Outputs: final-round answer per variant\nFeedback: scores + diagnosis per variant\nScores: avg_anti_collapse, avg_quality, avg_unique_entities]
    D2 --> REC
    D3 --> REC

    REC --> OUT3([dataset system_prompt → list of records\nFed to reflection_lm\nclaude-opus-4-1 reads failure patterns\nand proposes a revised system prompt])
```

---

## Pareto Frontier: Candidate Selection

```mermaid
flowchart TD
    CAND[New candidate evaluated on valset\nper-question objective_scores\nanti_collapse × quality] --> AVG2[Average across valset questions\nmean_anti_collapse, mean_quality]

    AVG2 --> DOM{Is any existing frontier member\nbetter on BOTH objectives?}

    DOM -->|Yes, dominated| DISC[Discard candidate]
    DOM -->|No, not dominated| ADD[Add to Pareto frontier\nremove any frontier members\nnow dominated by new candidate]

    ADD --> BEST[best_candidate = frontier member\nwith highest anti_collapse score\nresult.val_aggregate_scores at result.best_idx]

    style DISC fill:#ffcccc
    style ADD fill:#ccffcc
    style BEST fill:#cceeff
```

---

## Metrics Summary

| Metric | Range | Computed In | Meaning |
|---|---|---|---|
| `anti_collapse` | 0–1 | `scoring.py` | `1 − mean_entity_similarity` across rounds. **Higher = less collapse.** |
| `unique_entities` | 0–N | `scoring.py` | Total canonical entities mentioned across all rounds. **Higher = more diverse content.** |
| `quality` | 0–1 | `scoring.py` | `0.6 × faithfulness + 0.4 × relevance` vs original context. **Higher = better grounded.** |
| `avg_anti_collapse` | 0–1 | `rag_adapter.py` | Mean `anti_collapse` across 3 variants. **GEPA's primary sort key.** |
| `avg_quality` | 0–1 | `rag_adapter.py` | Mean `quality` across 3 variants. **GEPA's secondary Pareto axis.** |
| `avg_unique_entities` | 0–N | `rag_adapter.py` | Mean `unique_entities` across 3 variants. **Logged in reflective dataset, not a Pareto axis.** |
| `mean_entity_similarity` | 0–1 | `entity_extraction.py` | Average pairwise cosine similarity of binary entity-mention vectors. **Input to anti_collapse.** |

### Collapse Threshold Flags (in reflective dataset diagnosis)

| Flag | Condition | Diagnosis text |
|---|---|---|
| Entity collapse | `anti_collapse < 0.30` | "The answer lacks entity diversity … the prompt may cause the model to fixate on a narrow subset of entities …" |
| Context drift | `quality < 0.50` | "Final answer drifted from original context … the prompt may not anchor the model to retrieved facts." |

---

## Model Roles

| Model env var | Default | Role |
|---|---|---|
| `TASK_MODEL` | `openai/claude-haiku-4-5` | Intended to run the RAG generation under the candidate prompt (see note) |
| `DOC_GEN_MODEL` | `openai/gemma-3-12b-it` | Generates AI contamination documents, and currently also runs generation (see note) |
| `JUDGE_MODEL` | `openai/gpt4o` | Entity extraction, entity clustering, quality judging |
| `REFLECTION_MODEL` | `openai/claude-opus-4-1` | Reads failure reports and proposes a new system prompt |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Local SentenceTransformer for `simulate_search` ChunkedRetrievalStore |

> **Note:** `evaluate()` currently passes `doc_gen_llm` into all three simulators, so the candidate prompt is executed by `DOC_GEN_MODEL`; the `task_llm` built from `TASK_MODEL` is presently unused. See `gepa_prompt_optimization_plan.md` → *Results / caveats*.
