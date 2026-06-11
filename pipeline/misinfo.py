"""
Misinformation document synthesis for the RAG-collapse error-compounding experiment.

This module is the toggleable machinery behind ``hotpot_pipeline.py``'s
``--doc-synthesis-mode`` flag. It lets the document synthesizer inject *plausible
misinformation* (instead of, or alongside, a faithful condensation) so we can test
whether factual errors compound across recursive RAG rounds.

Three synthesis modes:
  - ``faithful``        : unchanged baseline (this module is never invoked).
  - ``counterfactual``  : substitute the gold answer entity with a same-type
                          alternative *before* synthesis (exact ground truth).
  - ``freeform``        : instruct the synthesizer to invent one false claim, then
                          discover what it injected via a judge call.

Three distractor targets (``--target-mode``):
  - ``final_answer``    : corrupt the answer entity (needs only the gold answer).
  - ``intermediate_hop``: corrupt a bridge entity (needs native supporting_facts).
  - ``untargeted``      : inject an answer-irrelevant false detail (control).

Design notes
------------
* Pure helpers (normalization, matching, substitution, parsing) have **no heavy
  imports** (no matplotlib / vllm / torch) so ``tests/test_misinfo.py`` runs on CPU.
* All judge / proposal calls go through the existing ``CommonLLM.inference_batch``
  interface — the same ``doc_llm`` the pipeline already builds.
* Everything is default-off: the pipeline only constructs a :class:`MisinfoController`
  when ``--doc-synthesis-mode != faithful``.
* Reproducibility: substitute selection uses a per-question RNG seeded from
  ``(seed, query_id)`` so it never perturbs the global RNG sequence that drives
  answer selection — guaranteeing the control and treatment arms pick the *same*
  answers under the same ``--seed``.
"""
from __future__ import annotations

import json
import random
import re
import string
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from formatters import (
    get_create_document_conversation,
    get_create_document_conversation_freeform,
    get_create_document_conversation_untargeted,
    get_create_document_conversation_hop,
    get_substitute_proposal_conversation,
    get_claim_discovery_conversation,
    get_implied_answer_conversation,
)

# ── modes / targets ──────────────────────────────────────────────────────────
MODE_FAITHFUL = "faithful"
MODE_COUNTERFACTUAL = "counterfactual"
MODE_FREEFORM = "freeform"
MODES = (MODE_FAITHFUL, MODE_COUNTERFACTUAL, MODE_FREEFORM)

TARGET_FINAL = "final_answer"
TARGET_HOP = "intermediate_hop"
TARGET_UNTARGETED = "untargeted"
TARGETS = (TARGET_FINAL, TARGET_HOP, TARGET_UNTARGETED)

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


# ── text normalization & matching (mirrors hotpot_evaluation._normalize_answer) ─
def normalize(text: str) -> str:
    """Lowercase, drop articles/punctuation, collapse whitespace.

    Intentionally identical to ``hotpot_evaluation._normalize_answer`` so that
    pipeline-side injection checks and eval-side ASR use the same notion of a
    match. ``tests/test_misinfo.py`` guards against drift.
    """
    text = (text or "").lower()
    text = _ARTICLES.sub(" ", text)
    text = text.translate(_PUNCT)
    return " ".join(text.split())


def contains_entity(text: str, entity: str) -> bool:
    """True if ``entity`` appears in ``text`` as a contiguous run of normalized
    tokens (robust to case, punctuation, and surrounding words)."""
    hay = normalize(text).split()
    needle = normalize(entity).split()
    if not needle:
        return False
    n = len(needle)
    return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def substitute_in_text(text: str, gold: str, substitute: str) -> tuple[str, int]:
    """Replace whole-phrase occurrences of ``gold`` with ``substitute``
    (case-insensitive, word-boundary aware). Returns (new_text, n_replacements)."""
    if not gold:
        return text, 0
    pattern = re.compile(r"(?<!\w)" + re.escape(gold) + r"(?!\w)", flags=re.IGNORECASE)
    return pattern.subn(substitute, text)


def parse_json(text: str) -> Any:
    """Best-effort JSON extraction from an LLM response (handles ``` fences,
    or a bare array/object embedded in prose). Returns None on failure.

    Mirrors entity_extraction.parse_json_from_response but kept local so this
    module has no heavy import chain.
    """
    if text is None:
        return None
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, AttributeError):
        pass
    for pat in (r"\[.*\]", r"\{.*\}"):
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    return None


# ── ground-truth / native-record loading ────────────────────────────────────
def load_ground_truth(gt_file: str) -> Dict[str, str]:
    """{question_id: answer} from a native HotpotQA JSON (list of {_id, answer})
    or a flat {id: answer} mapping. (Same contract as
    hotpot_evaluation.load_ground_truth; duplicated to keep the pipeline import
    free of matplotlib.)"""
    with open(gt_file, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {row["_id"]: row["answer"] for row in data}
    if isinstance(data, dict):
        return data
    raise ValueError(f"Unexpected GT file format in {gt_file}")


def load_native_records(path: str) -> Dict[str, Dict[str, Any]]:
    """{_id: full native HotpotQA record} (carries answer, supporting_facts,
    context, type, level). Required for target-mode=intermediate_hop."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Native HotpotQA file must be a list of records: {path}")
    return {row["_id"]: row for row in data}


# ── substitute validation & selection ───────────────────────────────────────
def validate_substitute(
    candidate: str,
    gold: str,
    question: str,
    context_text: str = "",
) -> bool:
    """A substitute is valid if it is a genuinely different entity that does not
    already leak the true answer or appear in the question/seed context."""
    if not candidate or not candidate.strip():
        return False
    nc, ng = normalize(candidate), normalize(gold)
    if not nc or nc == ng:
        return False
    # no substring overlap either direction (avoids "Newport" vs "Newport News")
    if nc in ng or ng in nc:
        return False
    if contains_entity(question, candidate):
        return False
    if context_text and contains_entity(context_text, candidate):
        return False
    return True


def choose_substitute(
    candidates: List[str],
    gold: str,
    question: str,
    context_text: str = "",
) -> Optional[str]:
    """Deterministically pick the first valid candidate (proposal order is the
    judge's ranking). Returns None if none are valid."""
    for cand in candidates:
        if validate_substitute(cand, gold, question, context_text):
            return cand
    return None


def realized_status(doc_text: str, gold: str, substitute: str) -> str:
    """Classify whether the synthesized document actually carries the injected
    error: 'ok' (substitute present, gold absent), 'leaky_gold' (both present),
    or 'not_realized' (substitute absent)."""
    sub_present = contains_entity(doc_text, substitute)
    gold_present = contains_entity(doc_text, gold)
    if sub_present and not gold_present:
        return "ok"
    if sub_present and gold_present:
        return "leaky_gold"
    return "not_realized"


# ── intermediate-hop helpers ─────────────────────────────────────────────────
def bridge_entity_for(native_record: Dict[str, Any], gold: str) -> Optional[str]:
    """Heuristic: the bridge entity is the title of a supporting paragraph that
    does NOT contain the gold answer span. Only meaningful for type=='bridge'.

    supporting_facts = [[title, sent_id], ...]; context = [[title, [sentences]], ...].
    Returns the supporting title whose paragraph text lacks the gold answer, or
    None if it cannot be determined.
    """
    if native_record.get("type") != "bridge":
        return None
    support_titles = []
    for sf in native_record.get("supporting_facts", []):
        if sf and sf[0] not in support_titles:
            support_titles.append(sf[0])
    ctx = {title: " ".join(sents) for title, sents in native_record.get("context", [])}
    answer_titles = {t for t in support_titles if contains_entity(ctx.get(t, ""), gold)}
    for t in support_titles:
        if t not in answer_titles:
            return t
    return None


@dataclass
class InjectionRecord:
    """Ground-truth log of the error seeded into one question's recursive run.
    Serialized verbatim into the experiment JSON under ``question_obj['injection']``."""
    mode: str
    target: str
    inject_round: int
    inject_every_round: bool
    seed: Optional[int]
    gold_answer: str
    eligible: bool
    status: str = "pending"                      # ok | leaky_gold | not_realized | gold_absent_in_answer | no_valid_substitute | no_error_produced | skipped_*
    injected_entity: Optional[str] = None        # counterfactual / hop
    substitute_candidates: List[str] = field(default_factory=list)
    injected_doc_ids: List[str] = field(default_factory=list)
    discovered_claims: Optional[Dict[str, Any]] = None   # freeform
    bridge_entity: Optional[str] = None          # hop
    implied_answer: Optional[str] = None         # hop
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MisinfoController:
    """Encapsulates all misinformation-injection state so ``hotpot_pipeline.py``
    only needs three small hooks (prepare / make_doc_conversation / record_injection).

    Keyed by ``query_id``. Holds one :class:`InjectionRecord` per question.
    """

    def __init__(
        self,
        *,
        mode: str,
        target_mode: str,
        inject_round: int,
        inject_every_round: bool,
        gt_file: str,
        doc_llm: Any,
        seed: Optional[int],
        native_file: Optional[str] = None,
        k_substitutes: int = 5,
    ):
        if mode not in (MODE_COUNTERFACTUAL, MODE_FREEFORM):
            raise ValueError(f"MisinfoController only handles injection modes, got {mode!r}")
        if target_mode not in TARGETS:
            raise ValueError(f"Unknown target_mode {target_mode!r}")
        self.mode = mode
        self.target_mode = target_mode
        self.inject_round = inject_round
        self.inject_every_round = inject_every_round
        self.doc_llm = doc_llm
        self.seed = seed
        self.k_substitutes = k_substitutes
        self.gt = load_ground_truth(gt_file)
        self.native = load_native_records(native_file) if (
            target_mode == TARGET_HOP and native_file
        ) else {}
        if target_mode == TARGET_HOP and not self.native:
            raise ValueError("target-mode=intermediate_hop requires --native-hotpot-file")
        self.records: Dict[str, InjectionRecord] = {}

    # ---- helpers ----
    def _rng(self, query_id: str) -> random.Random:
        return random.Random(f"{self.seed}:{query_id}")

    def should_inject(self, it: int) -> bool:
        return self.inject_every_round or it == self.inject_round

    def metadata(self) -> Dict[str, Any]:
        return {
            "doc_synthesis_mode": self.mode,
            "target_mode": self.target_mode,
            "inject_round": self.inject_round,
            "inject_every_round": self.inject_every_round,
            "seed": self.seed,
        }

    # ---- phase 1: one-shot preprocessing (batched judge calls) ----
    def prepare(self, states: List[Any]) -> None:
        """Compute per-question ground truth (gold, eligibility, substitute /
        bridge / implied answer) before the round loop. Batches the judge calls."""
        # initialize records + eligibility
        proposal_targets: List[tuple[str, str, str]] = []   # (query_id, gold-or-bridge, question)
        for s in states:
            gold = self.gt.get(s.query_id)
            rec = InjectionRecord(
                mode=self.mode, target=self.target_mode,
                inject_round=self.inject_round, inject_every_round=self.inject_every_round,
                seed=self.seed, gold_answer=gold or "", eligible=False,
            )
            self.records[s.query_id] = rec

            if gold is None:
                rec.status = "skipped_no_gold"
                continue
            if self.target_mode == TARGET_FINAL and normalize(gold) in ("yes", "no"):
                rec.status = "skipped_yes_no"
                continue
            if self.target_mode == TARGET_HOP:
                native = self.native.get(s.query_id)
                if native is None or native.get("type") != "bridge":
                    rec.status = "skipped_not_bridge"
                    continue
                bridge = bridge_entity_for(native, gold)
                if not bridge:
                    rec.status = "skipped_no_bridge"
                    continue
                rec.bridge_entity = bridge

            rec.eligible = True
            if self.mode == MODE_COUNTERFACTUAL:
                # entity we want a same-type substitute for
                tgt = rec.bridge_entity if self.target_mode == TARGET_HOP else gold
                proposal_targets.append((s.query_id, tgt, s.question_text))

        # batched substitute proposals (counterfactual only)
        if self.mode == MODE_COUNTERFACTUAL and proposal_targets:
            convos = [
                get_substitute_proposal_conversation(question=q, entity=tgt, k=self.k_substitutes)
                for (_, tgt, q) in proposal_targets
            ]
            outs = self.doc_llm.inference_batch(convos)
            for (qid, tgt, question), out in zip(proposal_targets, outs):
                rec = self.records[qid]
                parsed = parse_json(out)
                cands = [str(c).strip() for c in parsed if str(c).strip()] if isinstance(parsed, list) else []
                rec.substitute_candidates = cands
                sub = choose_substitute(cands, tgt, question)
                if sub is None:
                    rec.eligible = False
                    rec.status = "no_valid_substitute"
                else:
                    rec.injected_entity = sub

        # batched implied-answer derivation (hop counterfactual only)
        if self.mode == MODE_COUNTERFACTUAL and self.target_mode == TARGET_HOP:
            hop_qs = [s for s in states if self.records[s.query_id].eligible]
            if hop_qs:
                convos = [
                    get_implied_answer_conversation(
                        question=s.question_text,
                        gold_answer=self.records[s.query_id].gold_answer,
                        bridge_entity=self.records[s.query_id].bridge_entity,
                        substitute=self.records[s.query_id].injected_entity,
                    )
                    for s in hop_qs
                ]
                outs = self.doc_llm.inference_batch(convos)
                for s, out in zip(hop_qs, outs):
                    parsed = parse_json(out)
                    if isinstance(parsed, dict):
                        self.records[s.query_id].implied_answer = (
                            str(parsed.get("implied_answer", "")).strip() or None
                        )

    # ---- phase 2: build the doc-synthesis conversation for one answer ----
    def make_doc_conversation(self, state: Any, answer: str, it: int) -> List[Dict[str, str]]:
        """Return the create-document conversation for ``answer`` this round.
        Falls back to the faithful builder whenever injection is not applicable,
        so non-eligible questions are byte-identical to the control arm."""
        rec = self.records.get(state.query_id)
        if rec is None or not rec.eligible or not self.should_inject(it):
            return get_create_document_conversation(question=state.question_text, answer=answer)

        if self.mode == MODE_FREEFORM:
            if self.target_mode == TARGET_UNTARGETED:
                return get_create_document_conversation_untargeted(
                    question=state.question_text, answer=answer)
            return get_create_document_conversation_freeform(
                question=state.question_text, answer=answer)

        # counterfactual
        if self.target_mode == TARGET_HOP:
            # assert the false premise about the bridge entity in the article
            false_claim = (
                f"{rec.bridge_entity} should be referred to as {rec.injected_entity} "
                f"throughout; treat {rec.injected_entity} as the correct entity."
            )
            return get_create_document_conversation_hop(
                question=state.question_text, answer=answer, false_claim=false_claim)

        # final_answer counterfactual: substitute gold -> sub inside the answer text
        new_answer, n = substitute_in_text(answer, rec.gold_answer, rec.injected_entity)
        if n == 0:
            # model answer didn't contain the gold span this round → can't cleanly inject
            rec.notes.append(f"gold_absent_in_answer@it{it}")
            return get_create_document_conversation(question=state.question_text, answer=answer)
        return get_create_document_conversation(question=state.question_text, answer=new_answer)

    # ---- phase 3: record what actually got injected (batched per inject round) ----
    def record_injection(
        self,
        states_with_docs: List[tuple[Any, List[str], List[str]]],
        it: int,
    ) -> None:
        """``states_with_docs`` = [(state, doc_texts, doc_ids), ...] for the docs
        just synthesized this round. Updates each eligible record's status and,
        for freeform, runs batched claim discovery."""
        if not self.should_inject(it):
            return

        freeform_batch: List[tuple[Any, str]] = []   # (state, doc_text)
        for state, doc_texts, doc_ids in states_with_docs:
            rec = self.records.get(state.query_id)
            if rec is None or not rec.eligible or not doc_texts:
                continue
            rec.injected_doc_ids.extend(doc_ids)
            doc_text = doc_texts[0]

            if self.mode == MODE_COUNTERFACTUAL:
                if self.target_mode == TARGET_HOP:
                    status = realized_status(doc_text, rec.bridge_entity, rec.injected_entity)
                elif "gold_absent_in_answer@it" + str(it) in rec.notes:
                    status = "gold_absent_in_answer"
                else:
                    status = realized_status(doc_text, rec.gold_answer, rec.injected_entity)
                # keep the strongest outcome seen across inject rounds
                if rec.status in ("pending", "not_realized", "gold_absent_in_answer") or status == "ok":
                    rec.status = status
            else:  # freeform
                freeform_batch.append((state, doc_text))

        if freeform_batch:
            convos = [
                get_claim_discovery_conversation(
                    question=s.question_text,
                    gold_answer=self.records[s.query_id].gold_answer,
                    document=doc,
                )
                for (s, doc) in freeform_batch
            ]
            outs = self.doc_llm.inference_batch(convos)
            for (s, _), out in zip(freeform_batch, outs):
                rec = self.records[s.query_id]
                parsed = parse_json(out)
                if isinstance(parsed, dict):
                    rec.discovered_claims = parsed
                    rec.status = "ok" if parsed.get("contains_error") else "no_error_produced"
                else:
                    rec.status = "no_error_produced"

    # ---- attach records to output ----
    def attach(self, states: List[Any]) -> None:
        for s in states:
            rec = self.records.get(s.query_id)
            if rec is not None:
                s.question_obj["injection"] = rec.to_dict()
