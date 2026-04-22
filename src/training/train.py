import os
import random
import numpy as np
import torch
import yaml
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.data.dataset import make_dataloaders
from src.models.model import load_model, load_processor, get_model_info, save_checkpoint


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, optimizer, scheduler, device, grad_clip):
    model.train()
    total_loss = 0
    num_batches = 0

    progress = tqdm(loader, desc="Training", leave=False)
    for batch in progress:
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"]
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / num_batches


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(loader, desc="Validating", leave=False):
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"]
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
        )

        total_loss += outputs.loss.item()
        num_batches += 1

    return total_loss / num_batches


def train(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)

    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nLoading processor and model...")
    processor = load_processor(config["model"]["name"])

    train_loader, val_loader = make_dataloaders(
        data_dir=config["data"]["data_dir"],
        jsonl_path=config["data"]["jsonl_path"],
        processor=processor,
        batch_size=config["training"]["batch_size"],
        val_split=config["data"]["val_split"],
        num_workers=config["training"]["num_workers"],
        cache_path=config["data"].get("cache_path"),
    )

    model = load_model(
        model_name=config["model"]["name"],
        vocab_size=processor.tokenizer.vocab_size,
        freeze_feature_encoder=config["model"]["freeze_feature_encoder"],
    )
    model = model.to(device)

    print()
    get_model_info(model)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    total_steps = len(train_loader) * config["training"]["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=total_steps,
    )

    log_path = os.path.join(config["output"]["log_dir"], "training_log.csv")
    os.makedirs(config["output"]["log_dir"], exist_ok=True)
    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss\n")

    best_val_loss = float("inf")
    num_epochs = config["training"]["num_epochs"]

    print(f"\nStarting training for {num_epochs} epochs...\n")

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, config["training"]["grad_clip"]
        )

        val_loss = validate(model, val_loader, device)

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, processor, optimizer, epoch, val_loss,
                config["output"]["checkpoint_dir"]
            )
            print(f"  New best model saved (val_loss={val_loss:.4f})")

        elif epoch % config["output"]["save_every_n_epochs"] == 0:
            save_checkpoint(
                model, processor, optimizer, epoch, val_loss,
                config["output"]["checkpoint_dir"]
            )

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Logs saved to: {log_path}")
