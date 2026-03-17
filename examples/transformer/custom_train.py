"""
custom_train.py

使用指定重计算策略进行单独训练（无 eager baseline 对比）。
适用于快速验证某个策略的训练效果和 BLEU 分数。

用法::

    cd examples/transformer
    RECOMPUTE='{"6": 0}' python custom_train.py           # ATenIR 编译训练
    python custom_train.py --eager                         # 纯 eager（排查编译问题）
    # 或通过 run.sh:
    ./run.sh custom-train
"""
import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import Transformer
from data_utils import build_dataloaders, BOS_ID, EOS_ID, PAD_ID
from aten_recompute.core import CompilerBackend, inject_layer_tags

# ═══════════════════════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════════════════════

D_MODEL        = 256
NUM_HEADS      = 4
NUM_LAYERS     = 4
D_FF           = 1024
MAX_SEQ_LENGTH = 64
DROPOUT        = 0.15

BATCH_SIZE     = 64
NUM_EPOCHS     = 30
LR             = 5e-4
WARMUP_STEPS   = 400
LABEL_SMOOTHING = 0.1
GRAD_CLIP      = 1.0
SEED           = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

_BOLD = "═" * 72
_LINE = "─" * 72

STRATEGY_NAMES = {
    "0": "不重计算",
    "1": "全部重计算",
    "2": "按节点名称关键字",
    "3": "按层步长选择",
    "4": "按比例选择前 N% 层",
    "5": "按 ATen 算子类型",
    "6": "自动廉价重计算（链深度）",
    "7": "min-cut 最优重计算",
}


def _strategy_desc(cfg: dict) -> str:
    if not cfg:
        return "策略 0: 不重计算"
    key, val = next(iter(cfg.items()))
    name = STRATEGY_NAMES.get(str(key), "未知策略")
    param = f", 参数: {val}" if val is not None else ""
    return f"策略 {key}: {name}{param}"


# ═══════════════════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_weights(model: nn.Module):
    """Xavier 均匀初始化（Transformer 标准做法）。"""
    for name, p in model.named_parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        elif "bias" in name:
            nn.init.zeros_(p)


def build_lr_lambda(warmup_steps: int):
    """
    线性 warmup + inverse sqrt decay，Transformer 标准调度。
    前 warmup_steps 步线性升到 base_lr，之后按 sqrt(warmup/step) 衰减。
    """
    def lr_lambda(step):
        step = max(step, 1)
        warmup_factor = min(step / warmup_steps, 1.0)
        decay_factor = (warmup_steps / max(step, warmup_steps)) ** 0.5
        return warmup_factor * decay_factor
    return lr_lambda


def build_model(src_vocab_size: int, tgt_vocab_size: int) -> Transformer:
    return Transformer(
        src_vocab_size, tgt_vocab_size,
        D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF,
        MAX_SEQ_LENGTH, DROPOUT, padding_idx=PAD_ID,
    )


def save_checkpoint(model, tag: str, src_vocab_size: int, tgt_vocab_size: int,
                    optimizer=None, scheduler=None, epoch: int = None, best_val_loss: float = None):
    """保存模型与可选训练状态（optimizer/scheduler/epoch）。"""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "d_model": D_MODEL,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "d_ff": D_FF,
            "max_seq_length": MAX_SEQ_LENGTH,
            "dropout": DROPOUT,
            "padding_idx": PAD_ID,
        },
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

    path = CHECKPOINT_DIR / f"transformer_{tag}.pt"
    torch.save(ckpt, path)
    print(f"  模型已保存: {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)")


# ═══════════════════════════════════════════════════════════════════════════════
#  训练 & 评估
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    model.train()
    total_loss, total_tokens = 0.0, 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(
            output.contiguous().view(-1, output.size(-1)),
            tgt[:, 1:].contiguous().view(-1),
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        non_pad = (tgt[:, 1:] != PAD_ID).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad
    return total_loss / total_tokens


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        output = model(src, tgt[:, :-1])
        loss = criterion(
            output.contiguous().view(-1, output.size(-1)),
            tgt[:, 1:].contiguous().view(-1),
        )
        non_pad = (tgt[:, 1:] != PAD_ID).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad
    return total_loss / total_tokens


# ═══════════════════════════════════════════════════════════════════════════════
#  Greedy 解码 & BLEU
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def greedy_decode(model, src, max_len, device):
    model.eval()
    batch_size = src.size(0)

    src_padding_mask = torch.eq(src, model.padding_idx)
    src_embedded = model.dropout1(model.positional_encoding1(model.encoder_embedding(src)))
    enc_output = src_embedded
    for enc_layer in model.encoder_layers:
        enc_output = enc_layer(enc_output, src_padding_mask)

    tgt_ids = torch.full((batch_size, 1), BOS_ID, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        tgt_padding_mask = torch.eq(tgt_ids, model.padding_idx)
        tgt_len = tgt_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=device), diagonal=1
        ).bool()

        tgt_embedded = model.dropout2(
            model.positional_encoding2(model.decoder_embedding(tgt_ids))
        )
        dec_output = tgt_embedded
        for dec_layer in model.decoder_layers:
            dec_output = dec_layer(
                dec_output, enc_output, causal_mask,
                src_padding_mask, tgt_padding_mask,
            )

        logits = model.fc(dec_output[:, -1, :])
        next_token = logits.argmax(dim=-1, keepdim=True)
        next_token = next_token.masked_fill(finished.unsqueeze(1), PAD_ID)
        tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
        finished = finished | (next_token.squeeze(1) == EOS_ID)
        if finished.all():
            break

    return tgt_ids


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def compute_corpus_bleu(
    references: List[str], hypotheses: List[str], max_n: int = 4,
) -> float:
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    ref_len, hyp_len = 0, 0

    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        ref_len += len(ref_tokens)
        hyp_len += len(hyp_tokens)
        for n in range(1, max_n + 1):
            ref_ngrams = _ngrams(ref_tokens, n)
            hyp_ngrams = _ngrams(hyp_tokens, n)
            clipped = {ng: min(c, ref_ngrams.get(ng, 0)) for ng, c in hyp_ngrams.items()}
            clipped_counts[n - 1] += sum(clipped.values())
            total_counts[n - 1] += max(len(hyp_tokens) - n + 1, 0)

    log_precisions = 0.0
    for n in range(max_n):
        if total_counts[n] == 0 or clipped_counts[n] == 0:
            return 0.0
        log_precisions += math.log(clipped_counts[n] / total_counts[n])
    log_precisions /= max_n
    bp = min(1.0 - ref_len / hyp_len, 0.0) if hyp_len > 0 else -1e9
    return 100.0 * math.exp(bp + log_precisions)


def compute_test_bleu(model, test_loader, tgt_tok, device) -> float:
    all_refs, all_hyps = [], []
    for src, tgt in test_loader:
        src = src.to(device)
        pred_ids = greedy_decode(model, src, MAX_SEQ_LENGTH, device)
        for i in range(src.size(0)):
            pred = pred_ids[i].tolist()
            if EOS_ID in pred:
                pred = pred[:pred.index(EOS_ID)]
            pred = [t for t in pred if t not in (BOS_ID, PAD_ID)]
            hyp = tgt_tok.decode(pred)
            ref = tgt[i].tolist()
            ref = [t for t in ref if t not in (BOS_ID, EOS_ID, PAD_ID)]
            ref_text = tgt_tok.decode(ref)
            all_refs.append(ref_text)
            all_hyps.append(hyp)
    return compute_corpus_bleu(all_refs, all_hyps)


# ═══════════════════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eager", action="store_true",
                        help="纯 eager 训练（不 compile），用于排查编译问题")
    args = parser.parse_args()

    use_eager = args.eager

    os.environ.setdefault("RECOMPUTE_LOG_LEVEL", "WARNING")
    torch._dynamo.config.cache_size_limit = 64
    # torch.set_float32_matmul_precision("high")

    strategy_config = json.loads(os.getenv("RECOMPUTE", '{"6": 0}'))
    strat_key = next(iter(strategy_config), "0")

    mode_str = "Eager (无编译)" if use_eager else f"ATenIR 编译 (策略 {strat_key})"
    print(f"\n{_BOLD}")
    print("  ATenIR 自定义重计算训练")
    print(_BOLD)
    print(f"  设备:       {DEVICE}")
    print(f"  训练模式:   {mode_str}")
    if not use_eager:
        print(f"  重计算策略: {_strategy_desc(strategy_config)}")
    print(f"  模型:       {NUM_LAYERS}L-{D_MODEL}d-{NUM_HEADS}h-{D_FF}ff")
    print(f"  训练:       {NUM_EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}")
    print(f"  调度:       warmup={WARMUP_STEPS} steps, label_smoothing={LABEL_SMOOTHING}")
    print(_LINE)

    # ── 1. 数据 ───────────────────────────────────────────────────────────────
    print("\n[阶段 1/4] 构建数据管线")
    print(_LINE)

    train_loader, val_loader, test_loader, src_tok, tgt_tok = build_dataloaders(
        vocab_size=5000, max_len=MAX_SEQ_LENGTH, batch_size=BATCH_SIZE,
    )
    src_vocab_size = src_tok.get_vocab_size()
    tgt_vocab_size = tgt_tok.get_vocab_size()
    print(f"  src_vocab={src_vocab_size}, tgt_vocab={tgt_vocab_size}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=LABEL_SMOOTHING)

    # ── 2. 构建模型 ──────────────────────────────────────────────────────────
    print(f"\n[阶段 2/4] 构建模型")
    print(_LINE)

    seed_everything(SEED)
    model = build_model(src_vocab_size, tgt_vocab_size).to(DEVICE)
    init_weights(model)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  模型参数量: {param_count:.1f}M")

    if use_eager:
        # ── Eager 模式：不编译，直接训练 ──
        train_model = model
        print("  模式: eager (无 torch.compile)")
    else:
        # ── 编译模式：注入层级标签 + torch.compile ──
        enc_layers = [(layer, i) for i, layer in enumerate(model.encoder_layers)]
        dec_layers = [
            (layer, len(model.encoder_layers) + i)
            for i, layer in enumerate(model.decoder_layers)
        ]
        inject_layer_tags(enc_layers + dec_layers)

        backend = CompilerBackend(strategy_config=strategy_config, save_ir=False)
        train_model = torch.compile(model, backend=backend, dynamic=False)
        print("  模式: torch.compile + ATenIR")
        print("  (首次前向将触发编译，可能需要 30-60 秒 ...)")

    # ── 3. 训练 ───────────────────────────────────────────────────────────────
    print(f"\n[阶段 3/4] 训练 ({NUM_EPOCHS} epochs)")
    print(_LINE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(WARMUP_STEPS),
    )
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    t0 = time.time()

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(
            train_model, train_loader, criterion, optimizer, DEVICE,
            scheduler=scheduler,
        )
        val_loss = evaluate(train_model, val_loader, criterion, DEVICE)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, "best", src_vocab_size, tgt_vocab_size)
            improved = " *"

        cur_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        print(f"  Epoch {epoch + 1:2d}/{NUM_EPOCHS} | "
              f"train={train_loss:.4f} | val={val_loss:.4f} | "
              f"lr={cur_lr:.2e} | time={elapsed:.1f}s{improved}")

    total_time = time.time() - t0

    # ── 4. 评估 & 保存 ───────────────────────────────────────────────────────
    print(f"\n[阶段 4/4] 测试集 BLEU & 保存结果")
    print(_LINE)

    bleu = compute_test_bleu(model, test_loader, tgt_tok, DEVICE)
    print(f"  Test BLEU-4:    {bleu:.2f}")
    print(f"  Best val_loss:  {best_val_loss:.4f}")
    print(f"  Total time:     {total_time:.1f}s")

    tag = "eager" if use_eager else f"strat{strat_key}"
    save_checkpoint(model, tag, src_vocab_size, tgt_vocab_size)

    # 保存训练结果 JSON
    from aten_recompute.utils.save_ir import _default_ir_dir
    model_name = os.getenv("MODEL_NAME", "Transformer")
    out_dir = _default_ir_dir(model_name, subfolder="custom_training")
    results = {
        "mode": "eager" if use_eager else "compiled",
        "strategy": strategy_config if not use_eager else None,
        "strategy_desc": _strategy_desc(strategy_config) if not use_eager else "eager",
        "model_config": {
            "d_model": D_MODEL, "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS, "d_ff": D_FF,
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
        },
        "training_config": {
            "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE,
            "lr": LR, "warmup_steps": WARMUP_STEPS,
            "label_smoothing": LABEL_SMOOTHING,
            "grad_clip": GRAD_CLIP, "seed": SEED,
        },
        "results": {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "test_bleu": bleu,
            "total_time_s": round(total_time, 1),
        },
    }
    results_path = os.path.join(out_dir, f"custom_train_{tag}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{_BOLD}")
    print(f"  完成。结果已保存:")
    print(f"    训练结果: {results_path}")
    print(f"    模型:     checkpoints/transformer_{tag}.pt")
    print(f"    最佳模型: checkpoints/transformer_best.pt")
    print(f"{_BOLD}\n")


if __name__ == "__main__":
    main()
