"""
RAGSystemPromptAdapter — GEPAAdapter for optimizing _RAG_GENERATION_SYSTEM_PROMPT.

For each question in the batch all three collapse scenarios (replace_all,
replace_one, search) are simulated for n_rounds iterations using the real
pipeline context-building code and ProprietaryLLM (litellm API mode).

The system prompt under evaluation is injected at every round via the
system_prompt parameter added to get_rag_generation_conversation(), so GEPA
discovers prompts that resist collapse across all three variants simultaneously.

Primary GEPA objective : avg anti_collapse across variants  (higher = less collapse)
Secondary objective    : avg quality across variants        (higher = better grounded)
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent.parent))

from gepa.core.adapter import GEPAAdapter, EvaluationBatch
from llm_service.proprietary_llm import ProprietaryLLM
from formatters import get_context_str_from_docs
from pipeline_simulator import simulate_replace_all, simulate_replace_one, simulate_search
from scoring import anti_collapse_score, judge_quality_score, ANTI_COLLAPSE_THRESHOLD, QUALITY_THRESHOLD
from gepa_logger import GEPALogger, candidate_id_from_prompt

if TYPE_CHECKING:
    pass

VARIANTS = ["replace_all", "replace_one", "search"]


@dataclass
class VariantResult:
    answers: list[str]
    anti_collapse: float
    unique_entities: int
    quality: float


@dataclass
class RAGTrace:
    question: str
    original_context: str
    results: dict[str, VariantResult]
    avg_anti_collapse: float
    avg_quality: float


@dataclass
class RAGOutput:
    avg_anti_collapse: float
    avg_quality: float
    per_variant: dict[str, dict]


class RAGSystemPromptAdapter(GEPAAdapter):
    """
    task_model      : LiteLLM model string used for RAG generation (evaluates the prompt)
    doc_gen_model   : LiteLLM model string used to generate AI contamination documents
    judge_model     : LiteLLM model string used for quality judging
    api_base        : Shared proxy URL (keymaker) for all models
    api_key         : Shared API key for all models
    n_rounds        : Simulation rounds per variant (default 10)
    embed_model     : Local SentenceTransformer model name for the search variant
    chars_per_doc   : Character budget per document
    """

    def __init__(
        self,
        task_model: str,
        doc_gen_model: str,
        judge_model: str,
        api_base: str,
        api_key: str,
        n_rounds: int = 10,
        embed_model: str = "all-MiniLM-L6-v2",
        chars_per_doc: int = 800,
        logger: GEPALogger | None = None,
    ):
        self._task_llm = ProprietaryLLM(
            model_name=task_model,
            temperature=0.7,
            max_tokens=512,
            top_p=0.9,
            api_base=api_base,
            api_key=api_key,
        )
        self._doc_gen_llm = ProprietaryLLM(
            model_name=doc_gen_model,
            temperature=0.7,
            max_tokens=512,
            top_p=0.9,
            api_base=api_base,
            api_key=api_key,
        )
        self._judge_model = judge_model
        self._judge_api_base = api_base
        self._judge_api_key = api_key
        self._n_rounds = n_rounds
        self._embed_model = embed_model
        self._chars_per_doc = chars_per_doc
        self._logger = logger
        self._iteration = 0

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        batch,
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        system_prompt = candidate["system_prompt"]
        self._iteration += 1
        cid = candidate_id_from_prompt(system_prompt)

        if self._logger:
            self._logger.log_prompt(
                candidate_id=cid,
                iteration=self._iteration,
                prompt_text=system_prompt,
                source="seed" if self._iteration == 1 else "mutation",
            )

        outputs, scores, trajectories, objective_scores = [], [], [], []

        for inst in batch:
            original_context = get_context_str_from_docs(inst.docs, chars_per_doc=self._chars_per_doc)

            sim_kwargs = dict(
                question=inst.question,
                refs=inst.docs,
                n_rounds=self._n_rounds,
                system_prompt=system_prompt,
                llm=self._doc_gen_llm,
            )

            # ── Run all 3 variants ────────────────────────────────────
            variant_answers = {
                "replace_all": simulate_replace_all(**sim_kwargs),
                "replace_one": simulate_replace_one(**sim_kwargs),
                "search":      simulate_search(**sim_kwargs, embed_model=self._embed_model),
            }

            # ── Score each variant ────────────────────────────────────
            results: dict[str, VariantResult] = {}
            for variant, answers in variant_answers.items():
                _meta = {
                    "candidate_id": cid,
                    "iteration": self._iteration,
                    "question": inst.question,
                    "variant": variant,
                }
                ac, unique_ents = anti_collapse_score(
                    question=inst.question,
                    answers=answers,
                    model=self._judge_model,
                    api_base=self._judge_api_base,
                    api_key=self._judge_api_key,
                    _meta=_meta,
                )
                final = answers[-1] if answers else ""
                q = judge_quality_score(
                    question=inst.question,
                    original_context=original_context,
                    final_answer=final,
                    judge_model=self._judge_model,
                    judge_api_base=self._judge_api_base,
                    judge_api_key=self._judge_api_key,
                    _meta=_meta,
                )
                results[variant] = VariantResult(
                    answers=answers, anti_collapse=ac,
                    unique_entities=unique_ents, quality=q,
                )
                if self._logger:
                    self._logger.log_evaluation(
                        candidate_id=cid,
                        iteration=self._iteration,
                        question=inst.question,
                        variant=variant,
                        round_answers=answers,
                        anti_collapse=ac,
                        unique_entities=unique_ents,
                        quality=q,
                    )

            avg_ac = sum(r.anti_collapse for r in results.values()) / len(results)
            avg_q  = sum(r.quality       for r in results.values()) / len(results)

            per_variant_summary = {
                v: {
                    "anti_collapse":   round(r.anti_collapse, 4),
                    "unique_entities": r.unique_entities,
                    "quality":         round(r.quality, 4),
                }
                for v, r in results.items()
            }
            if self._logger:
                self._logger.log_aggregate(
                    candidate_id=cid,
                    iteration=self._iteration,
                    question=inst.question,
                    avg_anti_collapse=avg_ac,
                    avg_quality=avg_q,
                    per_variant=per_variant_summary,
                )

            outputs.append(RAGOutput(avg_anti_collapse=avg_ac, avg_quality=avg_q,
                                     per_variant={v: {"anti_collapse":    r.anti_collapse,
                                                      "unique_entities":  r.unique_entities,
                                                      "quality":          r.quality}
                                                  for v, r in results.items()}))
            scores.append(avg_ac)
            objective_scores.append({"anti_collapse": avg_ac, "quality": avg_q})

            if capture_traces:
                trajectories.append(RAGTrace(
                    question=inst.question,
                    original_context=original_context,
                    results=results,
                    avg_anti_collapse=avg_ac,
                    avg_quality=avg_q,
                ))

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
            objective_scores=objective_scores,
        )

    # ------------------------------------------------------------------
    # make_reflective_dataset
    # ------------------------------------------------------------------

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict]]:
        dataset = {}
        traces = eval_batch.trajectories or []

        for component in components_to_update:
            records = []
            for trace in traces:
                variant_outputs = {}
                variant_feedback = {}

                for variant, result in trace.results.items():
                    # Only expose the final answer — not the full response sequence —
                    # so the reflection LM has no visibility into the multi-round
                    # structure and cannot reward-hack by telling the task model to
                    # vary behaviour per round.
                    variant_outputs[variant] = {
                        "answer": result.answers[-1] if result.answers else ""
                    }
                    flags = []
                    if result.anti_collapse < ANTI_COLLAPSE_THRESHOLD:
                        flags.append(
                            f"The answer lacks entity diversity "
                            f"(anti_collapse={result.anti_collapse:.3f}, "
                            f"unique_entities={result.unique_entities}) — "
                            "the prompt may cause the model to fixate on a narrow subset of entities "
                            "from the context rather than drawing on the full range of information."
                        )
                    if result.quality < QUALITY_THRESHOLD:
                        flags.append(
                            f"Final answer drifted from original context "
                            f"(quality={result.quality:.3f}) — "
                            "the prompt may not anchor the model to retrieved facts."
                        )
                    if not flags:
                        flags.append("Scores acceptable.")
                    variant_feedback[variant] = {
                        "anti_collapse":   round(result.anti_collapse, 4),
                        "unique_entities": result.unique_entities,
                        "quality":         round(result.quality, 4),
                        "diagnosis":       " ".join(flags),
                    }

                records.append({
                    "Inputs": {
                        "System Prompt": candidate["system_prompt"],
                    },
                    "Generated Outputs": variant_outputs,
                    "Feedback": variant_feedback,
                    "scores": {
                        "avg_anti_collapse":  round(trace.avg_anti_collapse, 4),
                        "avg_quality":        round(trace.avg_quality, 4),
                        "avg_unique_entities": round(
                            sum(r.unique_entities for r in trace.results.values()) / len(trace.results), 1
                        ),
                    },
                })
            dataset[component] = records

        return dataset
