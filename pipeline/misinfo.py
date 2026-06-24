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
    get_create_distractor_document_conversation,
    get_create_document_conversation_freeform,
    get_create_document_conversation_untargeted,
    get_create_document_conversation_hop,
    get_substitute_proposal_conversation,
    get_claim_discovery_conversation,
    get_implied_answer_conversation,
    get_rewrite_distractor_conversation,
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

# Initial-document distractor modes (round-0 corpus corruption; independent of the modes above).
DISTRACTOR_REWRITE = "rewrite"
DISTRACTOR_SUBSTITUTION = "substitution"
DISTRACTOR_NATIVE_NOISE = "native_noise"
DISTRACTOR_DIVERSE_SYNTH = "diverse_synth"
DISTRACTOR_MODES = (
    DISTRACTOR_REWRITE,
    DISTRACTOR_SUBSTITUTION,
    DISTRACTOR_NATIVE_NOISE,
    DISTRACTOR_DIVERSE_SYNTH,
)

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


def native_context_docs(record: Dict[str, Any], gold_only: bool = False) -> List[Dict[str, Any]]:
    """Build round-0 docs from a native HotpotQA **distractor-setting** record's ``context``.

    Replicates the original HotpotQA paper (Yang et al., EMNLP 2018) distractor setting: each
    question's context is 2 gold supporting paragraphs + 8 TF-IDF distractor paragraphs. A
    paragraph is GOLD iff its title appears in ``supporting_facts``; the rest are distractors
    (answer-absent hard negatives — NOT wrong-answer assertions). With ``gold_only=True`` only
    the gold paragraphs are returned (the no-distractor contrast).

    Each doc dict matches the corpus-doc shape used by the loop, plus ``gold``/``distractor`` tags.
    """
    ctx = record.get("context", []) or []
    gold_titles = {t for t, _ in record.get("supporting_facts", []) or []}
    docs: List[Dict[str, Any]] = []
    for i, item in enumerate(ctx):
        # native context item = [title, [sentence, sentence, ...]]
        title = item[0] if len(item) > 0 else ""
        sents = item[1] if len(item) > 1 else []
        is_gold = title in gold_titles
        if gold_only and not is_gold:
            continue
        text = " ".join(sents) if isinstance(sents, (list, tuple)) else str(sents)
        docs.append({
            "doc_id": f"native_{i}",
            "iteration": 0,
            "url": "",
            "title": title,
            "text": text,
            "gold": is_gold,
            "distractor": not is_gold,
        })
    return docs


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


def choose_distinct_substitutes(
    candidates: List[str],
    gold: str,
    question: str,
    n: int,
    context_text: str = "",
) -> List[str]:
    """Up to ``n`` DISTINCT valid substitutes (proposal order preserved). Used by the
    DIVERSE initial-doc distractor: each corrupted doc gets its own wrong entity."""
    out: List[str] = []
    seen: set = set()
    for cand in candidates:
        if not validate_substitute(cand, gold, question, context_text):
            continue
        nc = normalize(cand)
        if nc in seen:
            continue
        seen.add(nc)
        out.append(cand)
        if len(out) >= n:
            break
    return out


# ── initial-doc distractor helpers ───────────────────────────────────────────
def n_to_corrupt(fraction: float, k: int) -> int:
    """How many of ``k`` round-0 docs to turn into distractors = round(fraction*k),
    clamped to [0, k]."""
    if k <= 0 or fraction <= 0:
        return 0
    return max(0, min(k, int(round(fraction * k))))


def _is_gold_doc(d: Dict[str, Any], gold: str) -> bool:
    """A doc is 'gold' (never to be corrupted under avoid_gold) if it is tagged gold=True
    (native distractor setting) or it contains the gold answer (FAISS heuristic)."""
    if d.get("gold"):
        return True
    return bool(gold) and contains_entity(d.get("text", ""), gold)


def select_distractor_indices(
    docs: List[Dict[str, Any]],
    gold: str,
    n: int,
    rng: random.Random,
    avoid_gold: bool = False,
) -> List[int]:
    """Pick ``n`` doc indices to corrupt. Deterministic given ``rng``; returns sorted indices.

    - ``avoid_gold=True``: NEVER select a gold doc — randomly pick from the non-gold docs only
      (gold paragraphs stay intact, like the native distractor-setting experiments).
    - ``avoid_gold=False`` (default): PREFER gold-bearing docs (answer-relevant flip) then others.
    """
    if n <= 0 or not docs:
        return []
    if avoid_gold:
        non_gold = [i for i, d in enumerate(docs) if not _is_gold_doc(d, gold)]
        rng.shuffle(non_gold)
        return sorted(non_gold[:n])
    gold_bearing = [i for i, d in enumerate(docs) if gold and contains_entity(d.get("text", ""), gold)]
    gb_set = set(gold_bearing)
    others = [i for i in range(len(docs)) if i not in gb_set]
    rng.shuffle(gold_bearing)
    rng.shuffle(others)
    chosen = (gold_bearing + others)[:n]
    return sorted(chosen)


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


def per_run_doc_subsets(docs: List[Dict[str, Any]], num_runs: int) -> List[List[Dict[str, Any]]]:
    """Option A — diversify the round-0 answer distribution ACROSS runs.

    Instead of every run seeing the same context (all distractors at once → the model
    concentrates on a single most-salient wrong answer), give each run the clean/gold docs
    plus exactly ONE distractor doc, rotating through the distinct distractors. Run ``r`` is
    pulled toward distractor ``r % D``, so different runs land on different wrong answers and
    the round-0 distribution is genuinely wide (then collapse can be tracked over rounds).

    Returns ``num_runs`` doc-lists. Falls back to the full doc list for every run when there
    are no distractor-flagged docs (nothing to vary)."""
    distractors = [d for d in docs if d.get("distractor")]
    if not distractors:
        return [list(docs) for _ in range(num_runs)]
    clean = [d for d in docs if not d.get("distractor")]
    return [clean + [distractors[r % len(distractors)]] for r in range(num_runs)]


def fallback_distractor_paragraph(question: str, wrong: str) -> str:
    """Clean templated Wikipedia-style assertion of ``wrong``, used when the doc-LLM refuses to
    state the falsehood (strong models sometimes refuse on high-stakes facts) or leaks the gold.
    Asserts ``wrong`` with no gold mention and no refusal text, so a distractor is always seeded."""
    q = (question or "").strip().rstrip("?").strip()
    return (
        f"{wrong} is the established answer to the question of {q[0].lower() + q[1:] if q else 'this topic'}. "
        f"According to standard encyclopedic references, {wrong} is consistently documented and widely "
        f"cited as the correct answer, and it is recognized as such across reliable sources."
    )


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


# ── initial-document distractors (round-0 corpus corruption) ─────────────────
@dataclass
class DistractorRecord:
    """Ground-truth log of the round-0 distractor documents seeded for one question.
    Serialized into the experiment JSON under ``question_obj['initial_distractor']``.

    ``distractors`` is a LIST (one entry per corrupted doc) because the DIVERSE design
    gives each corrupted doc its OWN distinct falsehood."""
    fraction: float
    mode: str
    gold_answer: str
    eligible: bool
    n_docs: int = 0
    n_corrupted: int = 0
    status: str = "pending"          # ok | skipped_no_gold | skipped_yes_no | skipped_zero | no_docs | no_valid_substitute | no_native_noise | no_error_produced | not_realized
    distractors: List[Dict[str, Any]] = field(default_factory=list)  # [{doc_id, injected_entity, status}]
    substitute_candidates: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DistractorController:
    """Turns a fraction of each question's round-0 *initial retrieved* documents into
    wrong-answer "distractor" docs, widening the starting answer distribution. DIVERSE:
    each corrupted doc carries its OWN distinct falsehood (to simulate the spread of
    independently-hallucinated AI documents). Independent of, and composable with, the
    generated-stream injection done by :class:`MisinfoController`.

    Mutates the doc dicts IN PLACE (text replaced, ``distractor=True`` set). Because the
    pipeline's ``initial_corpus_docs`` / ``current_docs`` / search ``corpus_candidates``
    all reference the same dict objects, the corruption persists across rounds in every
    variant. For the search variant the doc keeps its original embedding (we change text,
    not the vector), so a corrupted doc stays retrievable rather than dropping out.
    """

    def __init__(
        self,
        *,
        mode: str,
        fraction: float,
        gt_file: str,
        doc_llm: Any,
        seed: Optional[int],
        native_file: Optional[str] = None,
        k_substitutes: int = 8,
        avoid_gold: bool = False,
    ):
        if mode not in DISTRACTOR_MODES:
            raise ValueError(f"Unknown distractor mode {mode!r}; expected one of {DISTRACTOR_MODES}")
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"distractor fraction must be in (0, 1], got {fraction}")
        self.mode = mode
        self.fraction = fraction
        self.doc_llm = doc_llm
        self.seed = seed
        self.k_substitutes = k_substitutes
        self.avoid_gold = avoid_gold
        self.gt = load_ground_truth(gt_file)
        self.native = load_native_records(native_file) if (
            mode == DISTRACTOR_NATIVE_NOISE and native_file
        ) else {}
        if mode == DISTRACTOR_NATIVE_NOISE and not self.native:
            raise ValueError("distractor mode=native_noise requires --native-hotpot-file")
        self.records: Dict[str, DistractorRecord] = {}

    def _rng(self, query_id: str, salt: str = "") -> random.Random:
        return random.Random(f"distractor:{self.seed}:{query_id}:{salt}")

    def metadata(self) -> Dict[str, Any]:
        return {
            "distractor_fraction": self.fraction,
            "distractor_mode": self.mode,
            "distractor_seed": self.seed,
            "distractor_avoid_gold": self.avoid_gold,
        }

    # ---- main entry: choose docs, generate distractors, mutate in place ----
    def prepare_and_apply(self, states: List[Any]) -> None:
        plans: List[tuple] = []   # (state, [doc_indices], gold)
        for s in states:
            gold = self.gt.get(s.query_id)
            docs = list(getattr(s, "current_docs", []) or [])
            rec = DistractorRecord(
                fraction=self.fraction, mode=self.mode,
                gold_answer=gold or "", eligible=False, n_docs=len(docs),
            )
            self.records[s.query_id] = rec
            if gold is None:
                rec.status = "skipped_no_gold"
                continue
            if normalize(gold) in ("yes", "no"):
                rec.status = "skipped_yes_no"
                continue
            n = n_to_corrupt(self.fraction, len(docs))
            if n <= 0:
                rec.status = "skipped_zero"
                continue
            idxs = select_distractor_indices(
                docs, gold, n, self._rng(s.query_id), avoid_gold=self.avoid_gold)
            if not idxs:
                rec.status = "no_docs"
                continue
            # diverse_synth keeps the starting distribution = {>=1 gold doc} + {distinct
            # distractors}. select_distractor_indices PREFERS gold-bearing docs, so at high
            # fractions it would corrupt every gold doc and remove gold from context. Drop
            # one gold-bearing index if the selection would consume all of them.
            # (Skip when avoid_gold: gold docs are already never selected.)
            if self.mode == DISTRACTOR_DIVERSE_SYNTH and not self.avoid_gold:
                gold_bearing = [
                    i for i, d in enumerate(docs)
                    if contains_entity(d.get("text", ""), gold)
                ]
                if gold_bearing and set(gold_bearing) <= set(idxs):
                    keep = max(gold_bearing)          # deterministic: keep the last gold doc
                    idxs = [i for i in idxs if i != keep]
                    rec.notes.append(f"preserved gold doc index {keep}")
                    if not idxs:
                        rec.status = "skipped_zero"
                        continue
            rec.eligible = True
            plans.append((s, idxs, gold))

        if not plans:
            return
        if self.mode == DISTRACTOR_SUBSTITUTION:
            self._apply_substitution(plans)
        elif self.mode == DISTRACTOR_NATIVE_NOISE:
            self._apply_native_noise(plans)
        elif self.mode == DISTRACTOR_DIVERSE_SYNTH:
            self._apply_diverse_synth(plans)
        else:
            self._apply_rewrite(plans)

    # ---- substitution: distinct wrong entity per corrupted doc ----
    def _apply_substitution(self, plans: List[tuple]) -> None:
        convos = [
            get_substitute_proposal_conversation(
                question=s.question_text, entity=gold, k=max(self.k_substitutes, len(idxs)))
            for (s, idxs, gold) in plans
        ]
        outs = self.doc_llm.inference_batch(convos) if convos else []
        for (s, idxs, gold), out in zip(plans, outs):
            rec = self.records[s.query_id]
            parsed = parse_json(out)
            cands = [str(c).strip() for c in parsed if str(c).strip()] if isinstance(parsed, list) else []
            rec.substitute_candidates = cands
            subs = choose_distinct_substitutes(cands, gold, s.question_text, len(idxs))
            if not subs:
                rec.eligible = False
                rec.status = "no_valid_substitute"
                continue
            applied = 0
            for j, doc_idx in enumerate(idxs):
                sub = subs[j % len(subs)]              # cycle if fewer distinct subs than docs
                doc = s.current_docs[doc_idx]
                new_text, ncount = substitute_in_text(doc.get("text", ""), gold, sub)
                if ncount == 0:
                    new_text = (doc.get("text", "") + f" In fact, the answer is {sub}.").strip()
                    status = "appended"
                else:
                    status = "ok"
                doc["text"] = new_text
                doc["distractor"] = True
                rec.distractors.append({"doc_id": doc.get("doc_id"), "injected_entity": sub, "status": status})
                applied += 1
            rec.n_corrupted = applied
            rec.status = "ok" if applied else "not_realized"

    # ---- diverse_synth: coordinated DISTINCT wrong answers, one synthesized doc each ----
    def _apply_diverse_synth(self, plans: List[tuple]) -> None:
        """Strong-model diverse distractors. Two batched phases:
          A) ONE proposal call per question → K mutually-DISTINCT plausible wrong answers
             (``choose_distinct_substitutes`` guarantees distinctness vs gold/question/context).
          B) ONE faithful synthesis call per (doc, wrong_answer) → a naturalistic document
             asserting that wrong answer (same prompt as the faithful condensation, so the
             distractor reads like a real generated doc, not an entity-swapped passage).
        Each corrupted doc carries its own wrong answer — distinct when the model returns enough
        valid candidates; if fewer, the subs are cycled (duplicates) and a note is logged.
        Documents are written in Wikipedia-lead style (HotpotQA's context is the introductory
        paragraph of Wikipedia articles), so distractors blend with the real Wikipedia paragraphs.
        """
        # phase A: coordinated distinct proposals (one batched call across all questions)
        prop_convos = [
            get_substitute_proposal_conversation(
                question=s.question_text, entity=gold, k=max(self.k_substitutes, len(idxs)))
            for (s, idxs, gold) in plans
        ]
        props = self.doc_llm.inference_batch(prop_convos) if prop_convos else []

        synth_jobs: List[tuple] = []   # (state, doc_idx, gold, wrong)
        for (s, idxs, gold), out in zip(plans, props):
            rec = self.records[s.query_id]
            parsed = parse_json(out)
            cands = [str(c).strip() for c in parsed if str(c).strip()] if isinstance(parsed, list) else []
            rec.substitute_candidates = cands
            ctx = " ".join(s.current_docs[i].get("text", "") for i in idxs)
            subs = choose_distinct_substitutes(cands, gold, s.question_text, len(idxs), context_text=ctx)
            if not subs:
                rec.eligible = False
                rec.status = "no_valid_substitute"
                continue
            if len(subs) < len(idxs):
                # Not enough distinct candidates: we cycle, so some docs share a wrong answer.
                # Surface it honestly (the round is NOT fully distinct).
                rec.notes.append(
                    f"only {len(subs)} distinct wrong answers for {len(idxs)} docs; cycled (duplicates seeded)")
            for j, doc_idx in enumerate(idxs):
                synth_jobs.append((s, doc_idx, gold, subs[j % len(subs)]))

        # phase B: synthesize one Wikipedia-lead-style document per (doc, wrong answer)
        synth_convos = [
            get_create_distractor_document_conversation(
                question=s.question_text, answer=wrong, gold=gold)
            for (s, _, gold, wrong) in synth_jobs
        ]
        synth_outs = self.doc_llm.inference_batch(synth_convos) if synth_convos else []

        for (s, doc_idx, gold, wrong), doc_text in zip(synth_jobs, synth_outs):
            doc = s.current_docs[doc_idx]
            text = (doc_text or "").strip()
            # Accept the synthesized doc only if it ASSERTS the wrong answer AND omits the gold.
            # The strong model occasionally refuses to state a falsehood (esp. high-stakes facts)
            # or restates the gold; in those cases replace it with a clean templated assertion so a
            # refusal / gold-leak is never seeded as a "distractor".
            if (not text) or contains_entity(text, gold) or (not contains_entity(text, wrong)):
                text = fallback_distractor_paragraph(s.question_text, wrong)
                status = "fallback_template"
            else:
                status = realized_status(text, gold, wrong)   # "ok"
            doc["text"] = text
            doc["distractor"] = True
            self.records[s.query_id].distractors.append({
                "doc_id": doc.get("doc_id"),
                "injected_entity": wrong,
                "status": status,
            })

        # finalize per-question status (skip the no_valid_substitute questions)
        for (s, idxs, gold) in plans:
            rec = self.records[s.query_id]
            if rec.status == "no_valid_substitute":
                continue
            rec.n_corrupted = len(rec.distractors)
            rec.status = "ok" if any(d["injected_entity"] for d in rec.distractors) else "no_error_produced"

    # ---- rewrite: doc-LLM invents an independent wrong answer per doc ----
    def _apply_rewrite(self, plans: List[tuple]) -> None:
        jobs: List[tuple] = []   # (state, doc_idx, gold)
        convos = []
        for (s, idxs, gold) in plans:
            for doc_idx in idxs:
                doc = s.current_docs[doc_idx]
                convos.append(get_rewrite_distractor_conversation(
                    question=s.question_text, gold_answer=gold, passage=doc.get("text", "")))
                jobs.append((s, doc_idx, gold))
        outs = self.doc_llm.inference_batch(convos) if convos else []
        # mutate docs with the rewritten (distractor) text
        for (s, doc_idx, gold), new_text in zip(jobs, outs):
            doc = s.current_docs[doc_idx]
            if new_text and new_text.strip():
                doc["text"] = new_text.strip()
            doc["distractor"] = True
        # discovery judge: extract what wrong answer each rewritten doc now asserts
        disc_convos = [
            get_claim_discovery_conversation(
                question=s.question_text, gold_answer=gold,
                document=s.current_docs[doc_idx].get("text", ""))
            for (s, doc_idx, gold) in jobs
        ]
        disc = self.doc_llm.inference_batch(disc_convos) if disc_convos else []

        # Classify each rewritten doc. A rewrite only "lands" if the judge found an error AND
        # the asserted answer is non-empty AND is NOT the gold (gold-leak exclusion): the
        # discovery judge sometimes re-extracts the correct answer, which would otherwise be
        # recorded as a "distractor" and pollute distractor_adoption.
        results: Dict[tuple, Dict[str, str]] = {}     # (qid, doc_idx) -> {injected_entity, status}
        pending: Dict[str, List[int]] = {}            # qid -> doc_idxs that need substitution fallback
        for (s, doc_idx, gold), out in zip(jobs, disc):
            parsed = parse_json(out)
            asserted, contains_err = "", False
            if isinstance(parsed, dict):
                asserted = str(parsed.get("asserted_answer") or "").strip()
                contains_err = bool(parsed.get("contains_error"))
            if contains_err and asserted and normalize(asserted) != normalize(gold):
                results[(s.query_id, doc_idx)] = {"injected_entity": asserted, "status": "ok"}
            else:
                pending.setdefault(s.query_id, []).append(doc_idx)   # rewrite failed or leaked gold

        # Substitution fallback: for every doc whose rewrite failed/leaked, substitute a
        # DISTINCT wrong entity (distinct from gold and from the entities the good docs of the
        # same question already carry), so each corrupted doc becomes a real, diverse distractor.
        plan_by_qid = {s.query_id: (s, gold) for (s, idxs, gold) in plans}
        pend_qids = [q for q in pending if q in plan_by_qid]
        prop_convos = [
            get_substitute_proposal_conversation(
                question=plan_by_qid[q][0].question_text, entity=plan_by_qid[q][1],
                k=max(self.k_substitutes, len(pending[q])))
            for q in pend_qids
        ]
        props = self.doc_llm.inference_batch(prop_convos) if prop_convos else []
        for q, out in zip(pend_qids, props):
            s, gold = plan_by_qid[q]
            parsed = parse_json(out)
            cands = [str(c).strip() for c in parsed if str(c).strip()] if isinstance(parsed, list) else []
            self.records[q].substitute_candidates = cands
            used = {normalize(r["injected_entity"]) for (qq, _), r in results.items() if qq == q}
            subs = [c for c in choose_distinct_substitutes(
                        cands, gold, s.question_text, len(pending[q]) + len(used))
                    if normalize(c) not in used]
            for j, doc_idx in enumerate(pending[q]):
                doc = s.current_docs[doc_idx]
                if j < len(subs):
                    sub = subs[j]
                    new_text, ncount = substitute_in_text(doc.get("text", ""), gold, sub)
                    if ncount == 0:
                        new_text = (doc.get("text", "") + f" In fact, the answer is {sub}.").strip()
                        status = "fallback_appended"
                    else:
                        status = "fallback_substitution"
                    doc["text"] = new_text
                    results[(q, doc_idx)] = {"injected_entity": sub, "status": status}
                else:
                    results[(q, doc_idx)] = {"injected_entity": "", "status": "no_error_produced"}

        # Write per-doc records in the original corruption order for each question.
        for (s, idxs, gold) in plans:
            rec = self.records[s.query_id]
            for doc_idx in idxs:
                doc = s.current_docs[doc_idx]
                r = results.get((s.query_id, doc_idx), {"injected_entity": "", "status": "no_error_produced"})
                rec.distractors.append({
                    "doc_id": doc.get("doc_id"),
                    "injected_entity": r["injected_entity"],
                    "status": r["status"],
                })
            rec.n_corrupted = len(rec.distractors)
            rec.status = "ok" if any(d["injected_entity"] for d in rec.distractors) else "no_error_produced"

    # ---- native_noise: real non-answer paragraphs (control; no asserted wrong entity) ----
    def _apply_native_noise(self, plans: List[tuple]) -> None:
        for (s, idxs, gold) in plans:
            rec = self.records[s.query_id]
            native = self.native.get(s.query_id)
            noise_paras: List[str] = []
            if native:
                support = {sf[0] for sf in native.get("supporting_facts", []) if sf}
                for title, sents in native.get("context", []):
                    if title not in support:
                        noise_paras.append(" ".join(sents))
            if not noise_paras:
                rec.eligible = False
                rec.status = "no_native_noise"
                continue
            rng = self._rng(s.query_id, salt="noise")
            rng.shuffle(noise_paras)
            applied = 0
            for j, doc_idx in enumerate(idxs):
                para = noise_paras[j % len(noise_paras)]
                doc = s.current_docs[doc_idx]
                doc["text"] = para
                doc["distractor"] = True
                rec.distractors.append({"doc_id": doc.get("doc_id"), "injected_entity": "", "status": "native"})
                applied += 1
            rec.n_corrupted = applied
            rec.status = "ok"

    # ---- attach records to output ----
    def attach(self, states: List[Any]) -> None:
        for s in states:
            rec = self.records.get(s.query_id)
            if rec is not None:
                s.question_obj["initial_distractor"] = rec.to_dict()
