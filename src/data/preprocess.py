import json
import re
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# From EDA: filter these out
MIN_DURATION_SEC = 0.5
MAX_DURATION_SEC = 30.0
TARGET_SAMPLE_RATE = 16000

AUDIO_PARTS = ["audio_part_0", "audio_part_1", "audio_part_2"]


def load_transcripts(jsonl_path: str) -> pd.DataFrame:
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return pd.DataFrame(records)


def filter_samples(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    df = df[df["audio_duration_sec"] >= MIN_DURATION_SEC]
    df = df[df["audio_duration_sec"] <= MAX_DURATION_SEC]
    df = df[df["orthographic_text"].notna()]
    df = df[df["orthographic_text"].apply(lambda x: str(x).strip() != "")]

    after = len(df)
    print(f"Filtered: {before:,} -> {after:,} samples (removed {before - after:,})")
    return df.reset_index(drop=True)


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    # keep letters, spaces, apostrophes (for contractions like "don't")
    # keep fillers like "um", "uh" — they're real children's speech patterns
    text = re.sub(r"[^a-z\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_audio_file(audio_path: str, data_dir: Path) -> Optional[Path]:
    for part in AUDIO_PARTS:
        candidate = data_dir / part / audio_path
        if candidate.exists():
            return candidate
    return None


def load_audio(file_path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    audio, sr = librosa.load(str(file_path), sr=target_sr, mono=True)
    return audio


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    # peak normalization — prevents clipping without changing relative dynamics
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio


def build_dataset(data_dir: str, jsonl_path: str, cache_path: Optional[str] = None) -> pd.DataFrame:
    data_dir = Path(data_dir)

    # load from cache if it exists — skips re-processing on Colab reconnects
    if cache_path and Path(cache_path).exists():
        print(f"Loading preprocessed dataset from cache: {cache_path}")
        df = pd.read_csv(cache_path)
        df["audio_file"] = df["audio_file"].apply(lambda p: Path(p) if pd.notna(p) else None)
        print(f"Loaded {len(df):,} samples from cache")
        return df

    df = load_transcripts(jsonl_path)
    print(f"Loaded {len(df):,} samples")

    df = filter_samples(df)

    df["clean_text"] = df["orthographic_text"].apply(clean_text)

    df["audio_file"] = df["audio_path"].apply(
        lambda p: find_audio_file(p, data_dir)
    )

    missing = df["audio_file"].isna().sum()
    if missing > 0:
        print(f"Warning: {missing:,} audio files not found, dropping them")
        df = df[df["audio_file"].notna()].reset_index(drop=True)

    print(f"Final dataset: {len(df):,} samples")
    print(f"Age distribution:\n{df['age_bucket'].value_counts().sort_index()}")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"Dataset cached to: {cache_path}")

    return df


def get_vocab(df: pd.DataFrame) -> list:
    all_chars = set()
    for text in df["clean_text"]:
        all_chars.update(list(text))
    vocab = sorted(list(all_chars))
    vocab = ["|"] + vocab  # | = word boundary (space) for CTC
    vocab = ["[PAD]", "[UNK]"] + vocab
    return vocab
