"""Shared training utilities for transformer example scripts."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def seed_everything(seed: int, deterministic: bool = True, benchmark: bool = False) -> None:
    """Set random seeds and cudnn behavior for reproducibility control."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


def save_checkpoint_file(
    path: Path,
    model: nn.Module,
    model_config: dict,
    optimizer=None,
    scheduler=None,
    epoch: Optional[int] = None,
    best_val_loss: Optional[float] = None,
) -> None:
    """Save model checkpoint with optional training states."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
    }

    if optimizer is not None:
        try:
            ckpt["optimizer_state_dict"] = optimizer.state_dict()
        except Exception:
            pass
    if scheduler is not None:
        try:
            ckpt["scheduler_state_dict"] = scheduler.state_dict()
        except Exception:
            pass
    if epoch is not None:
        ckpt["epoch"] = int(epoch)
    if best_val_loss is not None:
        try:
            ckpt["best_val_loss"] = float(best_val_loss)
        except Exception:
            pass

    torch.save(ckpt, path)
    print(f"  模型已保存: {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)")


def train_one_epoch_token_loss(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device,
    pad_id: int,
    grad_clip: float,
    scheduler=None,
    scaler=None,
    amp_enabled: bool = False,
    non_blocking: bool = False,
    set_to_none: bool = False,
) -> float:
    """Run one training epoch and return token-weighted loss."""
    model.train()
    total_loss, total_tokens = 0.0, 0

    for src, tgt in loader:
        src = src.to(device, non_blocking=non_blocking)
        tgt = tgt.to(device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=set_to_none)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(src, tgt[:, :-1])
            loss = criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt[:, 1:].contiguous().view(-1),
            )

        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        non_pad_tokens = (tgt[:, 1:] != pad_id).sum().item()
        total_loss += loss.item() * non_pad_tokens
        total_tokens += non_pad_tokens

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate_token_loss(
    model: nn.Module,
    loader,
    criterion,
    device,
    pad_id: int,
    amp_enabled: bool = False,
    non_blocking: bool = False,
) -> float:
    """Evaluate model and return token-weighted loss."""
    model.eval()
    total_loss, total_tokens = 0.0, 0

    for src, tgt in loader:
        src = src.to(device, non_blocking=non_blocking)
        tgt = tgt.to(device, non_blocking=non_blocking)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(src, tgt[:, :-1])
            loss = criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt[:, 1:].contiguous().view(-1),
            )

        non_pad_tokens = (tgt[:, 1:] != pad_id).sum().item()
        total_loss += loss.item() * non_pad_tokens
        total_tokens += non_pad_tokens

    return total_loss / max(total_tokens, 1)
