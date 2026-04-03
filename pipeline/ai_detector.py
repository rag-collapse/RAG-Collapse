"""
AI text detection using desklib/ai-text-detector-v1.01.

Scores a batch of text strings and returns LABEL_1 (AI-generated) probabilities.
Loaded lazily on first use; runs on CPU to avoid conflicting with the main LLM GPU.
"""

from typing import List


class AIDetector:
    """Thin wrapper around the desklib/ai-text-detector-v1.01 HuggingFace model."""

    MODEL_ID = "desklib/ai-text-detector-v1.01"

    def __init__(self, batch_size: int = 8) -> None:
        self._batch_size = batch_size
        self._pipe = None  # lazy init

    def _load(self) -> None:
        from transformers import pipeline as hf_pipeline

        self._pipe = hf_pipeline(
            "text-classification",
            model=self.MODEL_ID,
            device="cpu",
            truncation=True,
            batch_size=self._batch_size,
        )
        print(f"[AIDetector] Loaded {self.MODEL_ID} on CPU.", flush=True)

    def score_texts(self, texts: List[str]) -> List[float]:
        """
        Return LABEL_1 (AI-generated) probability for each text in [0.0, 1.0].
        Empty strings are given a score of 0.0 without being sent to the model.
        """
        if self._pipe is None:
            self._load()

        scores: List[float] = [0.0] * len(texts)

        # Collect non-empty texts and their original indices
        valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        if not valid_indices:
            return scores

        valid_texts = [texts[i] for i in valid_indices]
        results = self._pipe(valid_texts)

        for idx, entry in zip(valid_indices, results):
            if entry["label"] == "LABEL_1":
                scores[idx] = entry["score"]
            else:
                scores[idx] = 1.0 - entry["score"]

        return scores
