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
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import (
            AutoConfig,
            AutoTokenizer,
            AutoModelForSequenceClassification,
            pipeline as hf_pipeline,
        )

        # desklib/ai-text-detector-v1.01 was saved with an old transformers version.
        # Transformers 4.46+ added _get_key_renaming_mapping which rejects its state dict
        # as "corrupted". Bypass by loading weights manually and passing the instantiated
        # model directly to the pipeline (no from_pretrained on weights).
        config = AutoConfig.from_pretrained(self.MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        model = AutoModelForSequenceClassification.from_config(config)

        try:
            weights_path = hf_hub_download(self.MODEL_ID, "pytorch_model.bin")
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        except Exception:
            from safetensors.torch import load_file
            weights_path = hf_hub_download(self.MODEL_ID, "model.safetensors")
            state_dict = load_file(weights_path)

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        self._pipe = hf_pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
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
