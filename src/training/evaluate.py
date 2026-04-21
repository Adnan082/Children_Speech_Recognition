import torch
import yaml
import pandas as pd
from tqdm import tqdm
from jiwer import wer, cer

from src.data.dataset import make_dataloaders
from src.models.model import load_model, load_processor, load_checkpoint


def decode_predictions(logits, processor):
    predicted_ids = torch.argmax(logits, dim=-1)
    transcriptions = processor.batch_decode(predicted_ids)
    return transcriptions


@torch.no_grad()
def evaluate(model, loader, processor, device):
    model.eval()

    all_predictions = []
    all_references = []

    for batch in tqdm(loader, desc="Evaluating"):
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"]
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = model(input_values=input_values, attention_mask=attention_mask)

        predictions = decode_predictions(outputs.logits, processor)
        all_predictions.extend(predictions)

        labels = batch["labels"]
        labels[labels == -100] = processor.tokenizer.pad_token_id
        references = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        all_references.extend(references)

    return all_predictions, all_references


def compute_metrics(predictions, references):
    overall_wer = wer(references, predictions)
    overall_cer = cer(references, predictions)

    print(f"\n{'='*40}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*40}")
    print(f"Word Error Rate  (WER): {overall_wer:.4f} ({overall_wer*100:.2f}%)")
    print(f"Char Error Rate  (CER): {overall_cer:.4f} ({overall_cer*100:.2f}%)")

    return overall_wer, overall_cer


def evaluate_by_age(predictions, references, age_buckets):
    print(f"\n{'='*40}")
    print("WER BY AGE BUCKET")
    print(f"{'='*40}")

    results = {}
    df = pd.DataFrame({
        "prediction": predictions,
        "reference": references,
        "age_bucket": age_buckets,
    })

    for age in sorted(df["age_bucket"].unique()):
        subset = df[df["age_bucket"] == age]
        age_wer = wer(subset["reference"].tolist(), subset["prediction"].tolist())
        age_cer = cer(subset["reference"].tolist(), subset["prediction"].tolist())
        results[age] = {"wer": age_wer, "cer": age_cer}
        print(f"  Age {age}: WER={age_wer*100:.2f}%  CER={age_cer*100:.2f}%  (n={len(subset)})")

    return results


def print_samples(predictions, references, n=10):
    print(f"\n{'='*40}")
    print(f"SAMPLE PREDICTIONS (first {n})")
    print(f"{'='*40}")
    for i in range(min(n, len(predictions))):
        print(f"  Reference:  {references[i]}")
        print(f"  Predicted:  {predictions[i]}")
        print()


def run_evaluation(config_path: str = "configs/config.yaml", checkpoint_path: str = None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = load_processor(config["model"]["name"])

    _, val_loader = make_dataloaders(
        data_dir=config["data"]["data_dir"],
        jsonl_path=config["data"]["jsonl_path"],
        processor=processor,
        batch_size=config["training"]["batch_size"],
        val_split=config["data"]["val_split"],
        num_workers=config["training"]["num_workers"],
    )

    model = load_model(
        model_name=config["model"]["name"],
        vocab_size=processor.tokenizer.vocab_size,
        freeze_feature_encoder=config["model"]["freeze_feature_encoder"],
    )

    if checkpoint_path:
        from torch.optim import AdamW
        optimizer = AdamW(model.parameters(), lr=config["training"]["learning_rate"])
        model, _, _ = load_checkpoint(model, optimizer, checkpoint_path)

    model = model.to(device)

    predictions, references = evaluate(model, val_loader, processor, device)

    age_buckets = val_loader.dataset.df["age_bucket"].tolist()

    compute_metrics(predictions, references)
    evaluate_by_age(predictions, references, age_buckets)
    print_samples(predictions, references, n=10)
