"""
AI text detection using desklib/ai-text-detector-v1.01.

Scores a batch of text strings and returns LABEL_1 (AI-generated) probabilities.
Loaded lazily on first use. Runs on CUDA if available, falls back to CPU.
"""

import torch
import torch.nn as nn
from typing import List
from transformers import AutoConfig, AutoTokenizer, AutoModel, PreTrainedModel


class _DesklibModel(PreTrainedModel):
    """
    Custom architecture used by desklib/ai-text-detector-v1.01.
    The checkpoint stores a single linear classifier over attention-masked mean pooling,
    which does not match AutoModelForSequenceClassification's expected key layout.
    Loading via this class avoids the key-rename rejection in transformers 4.46+.
    """

    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden = outputs[0]
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_emb = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_emb / sum_mask
        return self.classifier(pooled)  # (B, 1) logits


class AIDetector:
    """Thin wrapper around desklib/ai-text-detector-v1.01. Uses GPU when available."""

    MODEL_ID = "desklib/ai-text-detector-v1.01"
    MAX_LEN = 768

    def __init__(self, batch_size: int = 32) -> None:
        self._batch_size = batch_size
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load(self) -> None:
        from huggingface_hub import hf_hub_download

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = AutoConfig.from_pretrained(self.MODEL_ID)
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)

        try:
            weights_path = hf_hub_download(self.MODEL_ID, "pytorch_model.bin")
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        except Exception:
            from safetensors.torch import load_file
            weights_path = hf_hub_download(self.MODEL_ID, "model.safetensors")
            state_dict = load_file(weights_path)

        model = _DesklibModel(config)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(self._device)
        self._model = model

        print(
            f"[AIDetector] Loaded {self.MODEL_ID} on {self._device}.",
            flush=True,
        )

    def score_texts(self, texts: List[str]) -> List[float]:
        """
        Return P(AI-generated) in [0.0, 1.0] for each text.
        Empty strings score 0.0 without hitting the model.
        """
        if self._model is None:
            self._load()

        scores: List[float] = [0.0] * len(texts)

        valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        if not valid_indices:
            return scores

        valid_texts = [texts[i] for i in valid_indices]

        batch_scores: List[float] = []
        for start in range(0, len(valid_texts), self._batch_size):
            batch = valid_texts[start : start + self._batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.MAX_LEN,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self._device)
            attention_mask = encoded["attention_mask"].to(self._device)

            with torch.no_grad():
                logits = self._model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(logits).squeeze(-1).cpu().tolist()

            if isinstance(probs, float):
                probs = [probs]
            batch_scores.extend(probs)

        for idx, score in zip(valid_indices, batch_scores):
            scores[idx] = score

        return scores
