import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import Wav2Vec2Processor
from src.data.preprocess import build_dataset, load_audio, normalize_audio


class ChildrenSpeechDataset(Dataset):
    def __init__(self, df, processor: Wav2Vec2Processor, sample_rate: int = 16000):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio = load_audio(row["audio_file"], target_sr=self.sample_rate)
        audio = normalize_audio(audio)

        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs.input_values.squeeze(0)

        labels = self.processor.tokenizer(row["clean_text"], return_tensors="pt")
        label_ids = labels.input_ids.squeeze(0)

        return {
            "input_values": input_values,
            "labels": label_ids,
            "age_bucket": row["age_bucket"],
            "utterance_id": row["utterance_id"],
        }


class SpeechCollator:
    def __init__(self, processor: Wav2Vec2Processor, padding: bool = True):
        self.processor = processor
        self.padding = padding

    def __call__(self, batch):
        input_values = [{"input_values": item["input_values"]} for item in batch]
        labels = [{"input_ids": item["labels"]} for item in batch]

        batch_inputs = self.processor.pad(
            input_values,
            padding=self.padding,
            return_tensors="pt",
        )

        batch_labels = self.processor.tokenizer.pad(
            labels,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding token id with -100 so CTC loss ignores it
        label_ids = batch_labels["input_ids"].masked_fill(
            batch_labels["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )

        return {
            "input_values": batch_inputs["input_values"],
            "attention_mask": batch_inputs.get("attention_mask"),
            "labels": label_ids,
        }


def make_sampler(df) -> WeightedRandomSampler:
    # from EDA: 8-11 has 77% of data — balance by age bucket
    age_counts = df["age_bucket"].value_counts()
    weights = df["age_bucket"].apply(lambda a: 1.0 / age_counts[a])
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights.values),
        num_samples=len(df),
        replacement=True,
    )
    return sampler


def make_dataloaders(
    data_dir: str,
    jsonl_path: str,
    processor: Wav2Vec2Processor,
    batch_size: int = 8,
    val_split: float = 0.1,
    num_workers: int = 0,
):
    df = build_dataset(data_dir, jsonl_path)

    # split into train/val — stratify by age bucket
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df, test_size=val_split, stratify=df["age_bucket"], random_state=42
    )
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")

    train_dataset = ChildrenSpeechDataset(train_df, processor)
    val_dataset = ChildrenSpeechDataset(val_df, processor)

    collator = SpeechCollator(processor)
    sampler = make_sampler(train_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
