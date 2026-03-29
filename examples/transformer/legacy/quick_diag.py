"""
quick_diag.py

在真实 Multi30k 数据上做 5 epoch 快速对比：Eager vs 策略 0 vs 策略 7
打印每 epoch 的 loss 和梯度范数，用于定位长训练 stall 的根因。
"""
import json
import os
import sys
import time
import copy

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn

from model import Transformer
from data_utils import build_dataloaders, BOS_ID, EOS_ID, PAD_ID
from aten_recompute.core import CompilerBackend, inject_layer_tags

torch.set_float32_matmul_precision("high")
torch._dynamo.config.cache_size_limit = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用与 custom_train.py 完全相同的配置
D_MODEL = 256
NUM_HEADS = 4
NUM_LAYERS = 4
D_FF = 1024
MAX_SEQ = 64
DROPOUT = 0.15
BATCH_SIZE = 64
LR = 5e-4
WARMUP_STEPS = 400
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 1.0
SEED = 42
NUM_EPOCHS = 5  # 只跑 5 epoch

_BOLD = "═" * 72
_LINE = "─" * 72


def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_lr_lambda(warmup_steps):
    def lr_lambda(step):
        step = max(step, 1)
        warmup_factor = min(step / warmup_steps, 1.0)
        decay_factor = (warmup_steps / max(step, warmup_steps)) ** 0.5
        return warmup_factor * decay_factor
    return lr_lambda


def grad_norm(model):
    total = 0.0
    count = 0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
            count += 1
    return total ** 0.5, count


def run_training(tag, model, train_loader, val_loader, criterion):
    """完整训练，打印每 epoch 的 loss + 梯度范数。"""
    print(f"\n  [{tag}]")
    print(f"  {'Epoch':>5}  {'Train':>10}  {'Val':>10}  {'GradNorm':>12}  "
          f"{'Params':>8}  {'LR':>10}")
    print(f"  {_LINE}")

    optimizer = torch.optim.Adam(
        # 始终用原始 model 的 parameters
        (model._orig_mod if hasattr(model, '_orig_mod') else model).parameters(),
        lr=LR, betas=(0.9, 0.98), eps=1e-9,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(WARMUP_STEPS),
    )

    results = []
    t0 = time.time()
    for epoch in range(NUM_EPOCHS):
        # ── Train ──
        model.train()
        total_loss, total_tokens = 0.0, 0
        last_gn, last_gc = 0.0, 0
        for src, tgt in train_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            loss = criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt[:, 1:].contiguous().view(-1),
            )
            loss.backward()
            last_gn, last_gc = grad_norm(
                model._orig_mod if hasattr(model, '_orig_mod') else model
            )
            nn.utils.clip_grad_norm_(
                (model._orig_mod if hasattr(model, '_orig_mod') else model).parameters(),
                max_norm=GRAD_CLIP,
            )
            optimizer.step()
            scheduler.step()
            non_pad = (tgt[:, 1:] != PAD_ID).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad

        train_loss = total_loss / total_tokens

        # ── Val ──
        model.eval()
        val_loss_sum, val_tokens = 0.0, 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                output = model(src, tgt[:, :-1])
                loss = criterion(
                    output.contiguous().view(-1, output.size(-1)),
                    tgt[:, 1:].contiguous().view(-1),
                )
                non_pad = (tgt[:, 1:] != PAD_ID).sum().item()
                val_loss_sum += loss.item() * non_pad
                val_tokens += non_pad
        val_loss = val_loss_sum / val_tokens

        cur_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        print(f"  {epoch + 1:5d}  {train_loss:10.4f}  {val_loss:10.4f}  "
              f"{last_gn:12.4f}  {last_gc:8d}  {cur_lr:10.2e}")
        results.append((train_loss, val_loss, last_gn))

    return results


def main():
    os.environ.setdefault("RECOMPUTE_LOG_LEVEL", "WARNING")

    print(f"\n{_BOLD}")
    print("  Multi30k 快速对比: Eager vs Compiled (5 epochs)")
    print(_BOLD)
    print(f"  设备: {DEVICE}")
    print(f"  模型: {NUM_LAYERS}L-{D_MODEL}d")
    print(f"  训练: {NUM_EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}")
    print(f"  调度: warmup={WARMUP_STEPS}, label_smoothing={LABEL_SMOOTHING}")
    print(_LINE)

    # ── 数据 ──
    train_loader, val_loader, _, src_tok, tgt_tok = build_dataloaders(
        vocab_size=5000, max_len=MAX_SEQ, batch_size=BATCH_SIZE,
    )
    src_vocab = src_tok.get_vocab_size()
    tgt_vocab = tgt_tok.get_vocab_size()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=LABEL_SMOOTHING)

    # ── 共享初始化 ──
    seed_everything(SEED)
    base_model = Transformer(
        src_vocab, tgt_vocab, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF,
        MAX_SEQ, DROPOUT, padding_idx=PAD_ID,
    ).to(DEVICE)
    for name, p in base_model.named_parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        elif "bias" in name:
            nn.init.zeros_(p)

    all_results = {}

    # ── 1. Eager ──
    print(f"\n{'─' * 30} Eager {'─' * 30}")
    eager_model = copy.deepcopy(base_model)
    all_results["eager"] = run_training(
        "Eager", eager_model, train_loader, val_loader, criterion,
    )
    del eager_model

    # ── 2. Compiled 策略 0 (不重计算) ──
    print(f"\n{'─' * 30} Compiled 策略 0 {'─' * 30}")
    torch._dynamo.reset()
    s0_model = copy.deepcopy(base_model)
    enc = [(l, i) for i, l in enumerate(s0_model.encoder_layers)]
    dec = [(l, len(s0_model.encoder_layers) + i)
           for i, l in enumerate(s0_model.decoder_layers)]
    inject_layer_tags(enc + dec)
    backend0 = CompilerBackend(strategy_config={"0": None}, save_ir=False)
    compiled_s0 = torch.compile(s0_model, backend=backend0, dynamic=False)
    all_results["strategy_0"] = run_training(
        "Compiled 策略 0", compiled_s0, train_loader, val_loader, criterion,
    )
    del s0_model, compiled_s0

    # ── 3. Compiled 策略 7 (min-cut) ──
    print(f"\n{'─' * 30} Compiled 策略 7 {'─' * 30}")
    torch._dynamo.reset()
    s7_model = copy.deepcopy(base_model)
    enc = [(l, i) for i, l in enumerate(s7_model.encoder_layers)]
    dec = [(l, len(s7_model.encoder_layers) + i)
           for i, l in enumerate(s7_model.decoder_layers)]
    inject_layer_tags(enc + dec)
    backend7 = CompilerBackend(strategy_config={"7": 0.5}, save_ir=False)
    compiled_s7 = torch.compile(s7_model, backend=backend7, dynamic=False)
    all_results["strategy_7"] = run_training(
        "Compiled 策略 7", compiled_s7, train_loader, val_loader, criterion,
    )
    del s7_model, compiled_s7

    # ── 4. 默认 torch.compile (无自定义 backend) ──
    print(f"\n{'─' * 30} 默认 torch.compile {'─' * 30}")
    torch._dynamo.reset()
    default_model = copy.deepcopy(base_model)
    compiled_default = torch.compile(default_model, dynamic=False)
    all_results["default_compile"] = run_training(
        "默认 torch.compile", compiled_default, train_loader, val_loader, criterion,
    )
    del default_model, compiled_default

    # ── 汇总 ──
    print(f"\n{_BOLD}")
    print("  5 Epoch 训练对比汇总")
    print(_BOLD)
    print(f"  {'Mode':<25s}  {'E1 train':>10s}  {'E5 train':>10s}  "
          f"{'E5 val':>10s}  {'E5 grad':>10s}  {'下降?':>6s}")
    print(f"  {_LINE}")
    for name, res in all_results.items():
        e1_train = res[0][0]
        e5_train = res[-1][0]
        e5_val = res[-1][1]
        e5_grad = res[-1][2]
        delta = e1_train - e5_train
        status = "OK" if delta > 0.1 else "STALL"
        print(f"  {name:<25s}  {e1_train:10.4f}  {e5_train:10.4f}  "
              f"{e5_val:10.4f}  {e5_grad:10.4f}  {status:>6s}")
    print(f"{_BOLD}\n")


if __name__ == "__main__":
    main()
