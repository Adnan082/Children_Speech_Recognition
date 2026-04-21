import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def load_model(model_name: str, vocab_size: int, freeze_feature_encoder: bool = True):
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name,
        ctc_loss_reduction="mean",
        pad_token_id=0,
        vocab_size=vocab_size,
        ignore_mismatched_sizes=True,
    )

    if freeze_feature_encoder:
        model.freeze_feature_encoder()

    return model


def load_processor(model_name: str):
    return Wav2Vec2Processor.from_pretrained(model_name)


def get_model_info(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {frozen:,}")
    print(f"Trainable:            {trainable/total*100:.1f}%")


def save_checkpoint(model, processor, optimizer, epoch, loss, checkpoint_dir: str):
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"Resumed from epoch {epoch}, loss {loss:.4f}")
    return model, optimizer, epoch
