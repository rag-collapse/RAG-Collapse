import json
import random
import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel, AutoModelForSequenceClassification
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "desklib": {
        "path": "desklib/ai-text-detector-v1.01",
        "type": "desklib",
        "max_len": 768,
    },
    "desklib-finetuned": {
        "path": "/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/hf_cache_oz",
        "type": "desklib",
        "max_len": 768,
    },
    "tmr": {
        "path": "Oxidane/tmr-ai-text-detector",
        "type": "sequence_classification",
        "max_len": 512,
    },
    "modernbert": {
        "path": "GeorgeDrayson/modernbert-ai-detection-raid-mage",
        "type": "sequence_classification",
        "max_len": 512,
    },
}

# ---------------------------------------------------------------------------
# Desklib custom architecture
# ---------------------------------------------------------------------------

class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())

        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output


# ---------------------------------------------------------------------------
# Detector wrappers — unified interface: predict_batch(texts, device) -> scores
# ---------------------------------------------------------------------------

class DesklibDetector:
    def __init__(self, model_path, max_len=768):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = DesklibAIDetectionModel.from_pretrained(model_path)
        self.max_len = max_len

    def to(self, device):
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()

    def predict_batch(self, texts, device):
        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            scores = torch.sigmoid(outputs["logits"]).squeeze(-1).cpu().tolist()

        if isinstance(scores, float):
            scores = [scores]
        return scores


class SequenceClassificationDetector:
    def __init__(self, model_path, max_len=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.max_len = max_len

    def to(self, device):
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()

    def predict_batch(self, texts, device):
        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            # label index 1 = AI-generated for both tmr and modernbert
            scores = probs[:, 1].cpu().tolist()

        if isinstance(scores, float):
            scores = [scores]
        return scores


def load_detector(name):
    cfg = MODEL_REGISTRY[name]
    if cfg["type"] == "desklib":
        return DesklibDetector(cfg["path"], max_len=cfg["max_len"])
    elif cfg["type"] == "sequence_classification":
        return SequenceClassificationDetector(cfg["path"], max_len=cfg["max_len"])
    else:
        raise ValueError(f"Unknown detector type: {cfg['type']}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def plot_score_histogram(human_scores, ai_scores, model_name, output_path):
    bins = np.linspace(0, 1, 41)  # 40 bins
    plt.figure(figsize=(10, 5))
    plt.hist(human_scores, bins=bins, alpha=0.5, color="blue", label=f"Human (n={len(human_scores)})")
    plt.hist(ai_scores,    bins=bins, alpha=0.5, color="red",  label=f"AI (n={len(ai_scores)})")
    plt.axvline(x=0.5, color="black", linestyle="--", linewidth=1, label="Threshold = 0.5")
    plt.xlabel("AI Detection Score")
    plt.ylabel("Count")
    plt.title(f"AI Detection Score Distribution — {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Histogram saved to {output_path}")


def evaluate_on_json(json_path, detector, device, model_name, threshold=0.5,
                     max_per_class=None, seed=42, batch_size=32,
                     output_file=None, histogram_file="score_histogram.png"):
    random.seed(seed)

    with open(json_path) as f:
        data = json.load(f)

    human_docs, ai_docs = [], []
    for question in data["questions"]:
        iters = {it["iteration_number"]: it for it in question["iterations"]}
        if 0 in iters:
            for doc in iters[0]["documents"]:
                if doc["doc_id"].startswith("corpus_"):
                    human_docs.append(doc["text"])
        if 1 in iters:
            for doc in iters[1]["documents"]:
                if doc["doc_id"].startswith("gen_"):
                    ai_docs.append(doc["text"])

    random.shuffle(human_docs)
    random.shuffle(ai_docs)

    if max_per_class is not None:
        human_docs = human_docs[:max_per_class]
        ai_docs = ai_docs[:max_per_class]

    texts = human_docs + ai_docs
    labels = [0] * len(human_docs) + [1] * len(ai_docs)

    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    texts, labels = list(texts), list(labels)

    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    log(f"Model: {model_name}")
    log(f"Evaluating {len(texts)} documents ({labels.count(1)} AI, {labels.count(0)} human)")

    detector.eval()
    scores = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting", unit="batch"):
        batch = texts[i:i + batch_size]
        scores.extend(detector.predict_batch(batch, device))

    preds = [1 if s >= threshold else 0 for s in scores]

    log("\n--- Classification Report ---")
    log(classification_report(labels, preds, target_names=["Human", "AI"]))

    human_scores = [s for s, label in zip(scores, labels) if label == 0]
    ai_scores    = [s for s, label in zip(scores, labels) if label == 1]
    plot_score_histogram(human_scores, ai_scores, model_name, output_path=histogram_file)

    # Print human documents that scored close to 1 (false positives)
    human_fp = sorted(
        [(s, t) for s, t, label in zip(scores, texts, labels) if label == 0 and s >= 0.9],
        key=lambda x: x[0], reverse=True
    )
    log(f"\n--- Human Documents Scored >= 0.9 (False Positives, n={len(human_fp)}) ---")
    for i, (score, text) in enumerate(human_fp[:20], 1):
        log(f"\n[{i}] Score: {score:.4f}")
        log(text[:500] + ("..." if len(text) > 500 else ""))

    human_tp = sorted(
        [(s, t) for s, t, label in zip(scores, texts, labels) if label == 0 and s <= 0.2],
        key=lambda x: x[0]
    )
    log(f"\n--- Human Documents Scored <= 0.2 (True Positives, n={len(human_tp)}) ---")
    for i, (score, text) in enumerate(human_tp[:20], 1):
        log(f"\n[{i}] Score: {score:.4f}")
        log(text[:500] + ("..." if len(text) > 500 else ""))

    if output_file:
        with open(output_file, "w") as f:
            f.write("\n".join(lines))
        print(f"Results saved to {output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="desklib",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Which detector model to use")
    parser.add_argument("--eval", type=str, default=None,
                        help="Path to example_run.json for evaluation")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Max documents per class (human/AI)")
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-file", type=str, default="ai_validation.txt",
                        help="File to save evaluation results")
    parser.add_argument("--histogram-file", type=str, default="score_histogram.png",
                        help="File to save score histogram")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {args.model}")
    detector = load_detector(args.model)
    detector.to(device)

    if args.eval:
        evaluate_on_json(
            args.eval, detector, device,
            model_name=args.model,
            threshold=args.threshold,
            max_per_class=args.max_per_class,
            seed=args.seed,
            batch_size=args.batch_size,
            output_file=args.output_file,
            histogram_file=args.histogram_file,
        )
    else:
        text_ai = "AI detection refers to the process of identifying whether a given piece of content, such as text, images, or audio, has been generated by artificial intelligence. This is achieved using various machine learning techniques, including perplexity analysis, entropy measurements, linguistic pattern recognition, and neural network classifiers trained on human and AI-generated data."
        text_human = "It is estimated that a major part of the content in the internet will be generated by AI / LLMs by 2025. This leads to a lot of misinformation and credibility related issues."

        detector.eval()
        scores = detector.predict_batch([text_ai, text_human], device)
        for label, score in zip(["AI sample", "Human sample"], scores):
            pred = "AI Generated" if score >= args.threshold else "Not AI Generated"
            print(f"[{label}] score={score:.4f} → {pred}")


if __name__ == "__main__":
    main()
