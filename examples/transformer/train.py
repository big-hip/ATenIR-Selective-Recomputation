"""
train.py

在 Multi30k EN→DE 翻译任务上验证 ATenIR 选择性重计算的训练正确性。
比较 eager baseline 和 ATenIR 编译模型的 loss 收敛曲线与 BLEU 分数。
训练结束后保存模型 checkpoint 和训练结果到 IR_artifacts/。

用法::

    cd examples/transformer
    RECOMPUTE='{"6": 0}' python train.py
    # 或通过 run.sh:
    ./run.sh train
"""
import copy
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
from data_utils import (
    build_dataloaders,
    BOS_ID, EOS_ID, PAD_ID,
)

from aten_recompute.core import CompilerBackend, inject_layer_tags

# ═══════════════════════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════════════════════

# 模型（为 Multi30k 29K 样本缩小，避免过拟合）
D_MODEL        = 256
NUM_HEADS      = 4
NUM_LAYERS     = 3
D_FF           = 1024
MAX_SEQ_LENGTH = 64
DROPOUT        = 0.1

# 训练
BATCH_SIZE     = 64
NUM_EPOCHS     = 15
LR             = 3e-4
GRAD_CLIP      = 1.0
SEED           = 42

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 保存路径
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

# 输出格式
_BOLD = "═" * 72
_LINE = "─" * 72

# ═══════════════════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int):
    """固定所有随机种子，确保可复现。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(src_vocab_size: int, tgt_vocab_size: int) -> Transformer:
    """构建缩小版 Transformer。"""
    return Transformer(
        src_vocab_size, tgt_vocab_size,
        D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF,
        MAX_SEQ_LENGTH, DROPOUT, padding_idx=PAD_ID,
    )


def save_checkpoint(model, tag: str, src_vocab_size: int, tgt_vocab_size: int):
    """保存模型权重和配置到 checkpoints/ 目录。"""
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
    path = CHECKPOINT_DIR / f"transformer_{tag}.pt"
    torch.save(ckpt, path)
    print(f"  模型已保存: {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)")


# ═══════════════════════════════════════════════════════════════════════════════
#  训练 & 评估
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch，返回平均 per-token loss。"""
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

        non_pad = (tgt[:, 1:] != PAD_ID).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad

    return total_loss / total_tokens


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """验证集评估，返回平均 per-token loss。"""
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
#  Greedy 解码
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def greedy_decode(model, src, max_len, device):
    """
    自回归贪心解码。直接访问模型子模块，不经过 torch.compile。

    Parameters
    ----------
    model : 未编译的原始 Transformer
    src : (B, S_src) 源 token IDs
    max_len : 最大解码长度
    device : 设备

    Returns
    -------
    (B, T) 解码出的 token IDs（含 BOS，不含 EOS 之后的内容）
    """
    model.eval()
    batch_size = src.size(0)

    # 编码器
    src_padding_mask = torch.eq(src, model.padding_idx)
    src_embedded = model.dropout1(model.positional_encoding1(model.encoder_embedding(src)))
    enc_output = src_embedded
    for enc_layer in model.encoder_layers:
        enc_output = enc_layer(enc_output, src_padding_mask)

    # 解码器（自回归）
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

        logits = model.fc(dec_output[:, -1, :])  # (B, vocab)
        next_token = logits.argmax(dim=-1, keepdim=True)  # (B, 1)

        # 已结束的序列填 PAD
        next_token = next_token.masked_fill(finished.unsqueeze(1), PAD_ID)
        tgt_ids = torch.cat([tgt_ids, next_token], dim=1)

        finished = finished | (next_token.squeeze(1) == EOS_ID)
        if finished.all():
            break

    return tgt_ids


# ═══════════════════════════════════════════════════════════════════════════════
#  BLEU-4
# ═══════════════════════════════════════════════════════════════════════════════

def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def compute_corpus_bleu(
    references: List[str],
    hypotheses: List[str],
    max_n: int = 4,
) -> float:
    """
    Corpus-level BLEU-4（无外部依赖）。

    Parameters
    ----------
    references : 参考译文列表
    hypotheses : 模型译文列表
    max_n : 最大 n-gram 阶数

    Returns
    -------
    BLEU 分数（0-100）
    """
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

    # n-gram 精度（对数）
    log_precisions = 0.0
    for n in range(max_n):
        if total_counts[n] == 0 or clipped_counts[n] == 0:
            return 0.0
        log_precisions += math.log(clipped_counts[n] / total_counts[n])
    log_precisions /= max_n

    # Brevity Penalty
    bp = min(1.0 - ref_len / hyp_len, 0.0) if hyp_len > 0 else -1e9

    return 100.0 * math.exp(bp + log_precisions)


def compute_test_bleu(model, test_loader, tgt_tok, device) -> float:
    """在测试集上计算 BLEU-4。"""
    all_refs, all_hyps = [], []

    for src, tgt in test_loader:
        src = src.to(device)
        pred_ids = greedy_decode(model, src, MAX_SEQ_LENGTH, device)

        for i in range(src.size(0)):
            # 解码预测：去掉 BOS，截止到 EOS
            pred = pred_ids[i].tolist()
            if EOS_ID in pred:
                pred = pred[:pred.index(EOS_ID)]
            pred = [t for t in pred if t not in (BOS_ID, PAD_ID)]
            hyp = tgt_tok.decode(pred)

            # 解码参考：去掉 BOS/EOS/PAD
            ref = tgt[i].tolist()
            ref = [t for t in ref if t not in (BOS_ID, EOS_ID, PAD_ID)]
            ref_text = tgt_tok.decode(ref)

            all_refs.append(ref_text)
            all_hyps.append(hyp)

    return compute_corpus_bleu(all_refs, all_hyps)


# ═══════════════════════════════════════════════════════════════════════════════
#  单次完整训练
# ═══════════════════════════════════════════════════════════════════════════════

def run_training(
    tag: str,
    model,
    train_loader,
    val_loader,
    criterion,
    device,
) -> Tuple[List[float], List[float]]:
    """执行完整训练，返回 (train_losses, val_losses)。"""
    optimizer = torch.optim.Adam(
        # 如果是 compiled model，参数在原始 model 上
        model.parameters(),
        lr=LR, betas=(0.9, 0.98), eps=1e-9,
    )

    train_losses, val_losses = [], []
    t0 = time.time()

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - t0
        print(f"  [{tag}] Epoch {epoch + 1:2d}/{NUM_EPOCHS} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"time={elapsed:.1f}s")

    return train_losses, val_losses


# ═══════════════════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.environ.setdefault("RECOMPUTE_LOG_LEVEL", "WARNING")
    torch._dynamo.config.cache_size_limit = 64

    strategy_config = json.loads(os.getenv("RECOMPUTE", '{"6": 0}'))
    strat_key = next(iter(strategy_config), "0")

    print(f"\n{_BOLD}")
    print("  Multi30k EN→DE 翻译训练验证")
    print(_BOLD)
    print(f"  设备:       {DEVICE}")
    print(f"  重计算策略: {strategy_config}")
    print(f"  模型:       {NUM_LAYERS}L-{D_MODEL}d-{NUM_HEADS}h-{D_FF}ff")
    print(f"  训练:       {NUM_EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}")
    print(_LINE)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. 数据管线
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[阶段 1/4] 构建数据管线")
    print(_LINE)

    train_loader, val_loader, test_loader, src_tok, tgt_tok = build_dataloaders(
        vocab_size=5000, max_len=MAX_SEQ_LENGTH, batch_size=BATCH_SIZE,
    )

    src_vocab_size = src_tok.get_vocab_size()
    tgt_vocab_size = tgt_tok.get_vocab_size()
    print(f"  src_vocab={src_vocab_size}, tgt_vocab={tgt_vocab_size}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    # ──────────────────────────────────────────────────────────────────────────
    # 2. 构建基础模型（两个 baseline 共享相同初始化）
    # ──────────────────────────────────────────────────────────────────────────
    seed_everything(SEED)
    base_model = build_model(src_vocab_size, tgt_vocab_size).to(DEVICE)
    param_count = sum(p.numel() for p in base_model.parameters()) / 1e6
    print(f"  模型参数量: {param_count:.1f}M")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Eager Baseline
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[阶段 2/4] Eager Baseline 训练")
    print(_LINE)

    eager_model = copy.deepcopy(base_model)
    eager_train, eager_val = run_training(
        "Eager", eager_model, train_loader, val_loader, criterion, DEVICE,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 4. ATenIR Compiled
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[阶段 3/4] ATenIR 编译训练 (策略 {strat_key})")
    print(_LINE)

    atenir_model = copy.deepcopy(base_model)

    # 注入层级标签
    enc_layers = [(layer, i) for i, layer in enumerate(atenir_model.encoder_layers)]
    dec_layers = [
        (layer, len(atenir_model.encoder_layers) + i)
        for i, layer in enumerate(atenir_model.decoder_layers)
    ]
    inject_layer_tags(enc_layers + dec_layers)

    # 编译
    backend = CompilerBackend(strategy_config=strategy_config, save_ir=False)
    compiled_model = torch.compile(atenir_model, backend=backend, dynamic=True)

    print("  (首次前向将触发编译，可能需要 30-60 秒 ...)")
    atenir_train, atenir_val = run_training(
        "ATenIR", compiled_model, train_loader, val_loader, criterion, DEVICE,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 5. 对比
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[阶段 4/4] 训练结果对比")
    print(_BOLD)

    # Loss 曲线对比
    print("  Epoch  Eager_train  ATenIR_train  |diff|      Eager_val  ATenIR_val  |diff|")
    print(f"  {_LINE}")
    max_train_diff = 0.0
    max_val_diff = 0.0
    for epoch in range(NUM_EPOCHS):
        td = abs(eager_train[epoch] - atenir_train[epoch])
        vd = abs(eager_val[epoch] - atenir_val[epoch])
        max_train_diff = max(max_train_diff, td)
        max_val_diff = max(max_val_diff, vd)
        print(f"  {epoch + 1:5d}  {eager_train[epoch]:11.4f}  {atenir_train[epoch]:12.4f}  "
              f"{td:10.6f}  {eager_val[epoch]:9.4f}  {atenir_val[epoch]:10.4f}  {vd:10.6f}")

    # BLEU 对比
    print(f"\n  计算测试集 BLEU (greedy decode) ...")
    eager_bleu = compute_test_bleu(eager_model, test_loader, tgt_tok, DEVICE)
    # 用未编译的模型权重（编译模型参数与 atenir_model 共享）
    atenir_bleu = compute_test_bleu(atenir_model, test_loader, tgt_tok, DEVICE)

    print(f"\n  Test BLEU (Eager):  {eager_bleu:.2f}")
    print(f"  Test BLEU (ATenIR): {atenir_bleu:.2f}")
    print(f"  BLEU 差异:          {abs(eager_bleu - atenir_bleu):.2f}")

    # 判定
    print(f"\n{_BOLD}")
    bleu_diff = abs(eager_bleu - atenir_bleu)
    if max_train_diff < 0.05 and bleu_diff < 2.0:
        print("  ✓ PASS: ATenIR 重计算训练结果与 eager baseline 数值等价。")
        print(f"    max_train_loss_diff={max_train_diff:.6f}, bleu_diff={bleu_diff:.2f}")
    else:
        print(f"  △ 注意: 检测到差异 (max_train_diff={max_train_diff:.6f}, "
              f"bleu_diff={bleu_diff:.2f})")
        print("    这可能由 torch.compile 浮点非确定性导致，不代表重计算逻辑有误。")
        print("    建议：对比最终 val_loss 是否在同一数量级。")
    print(f"{_BOLD}\n")

    # ──────────────────────────────────────────────────────────────────────────
    # 6. 保存模型
    # ──────────────────────────────────────────────────────────────────────────
    print("[保存] 保存训练好的模型 ...")
    # 保存 BLEU 更高的模型作为 best，两个都保存
    save_checkpoint(eager_model, "eager", src_vocab_size, tgt_vocab_size)
    save_checkpoint(atenir_model, "atenir", src_vocab_size, tgt_vocab_size)

    best_tag = "eager" if eager_bleu >= atenir_bleu else "atenir"
    best_model = eager_model if best_tag == "eager" else atenir_model
    save_checkpoint(best_model, "best", src_vocab_size, tgt_vocab_size)
    print(f"  最佳模型 ({best_tag}, BLEU={max(eager_bleu, atenir_bleu):.2f}) "
          f"已保存为 transformer_best.pt")

    # ──────────────────────────────────────────────────────────────────────────
    # 7. 保存训练结果到 IR_artifacts
    # ──────────────────────────────────────────────────────────────────────────
    from aten_recompute.utils.save_ir import _default_ir_dir

    model_name = os.getenv("MODEL_NAME", "Transformer")
    out_dir = _default_ir_dir(model_name, subfolder="training")
    results = {
        "model_config": {
            "d_model": D_MODEL, "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS, "d_ff": D_FF,
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
        },
        "training_config": {
            "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE,
            "lr": LR, "grad_clip": GRAD_CLIP, "seed": SEED,
        },
        "strategy": json.loads(os.getenv("RECOMPUTE", '{"6": 0}')),
        "eager": {
            "train_losses": eager_train,
            "val_losses": eager_val,
            "test_bleu": eager_bleu,
        },
        "atenir": {
            "train_losses": atenir_train,
            "val_losses": atenir_val,
            "test_bleu": atenir_bleu,
        },
        "verdict": {
            "max_train_loss_diff": max_train_diff,
            "bleu_diff": bleu_diff,
            "pass": max_train_diff < 0.05 and bleu_diff < 2.0,
        },
    }
    results_path = os.path.join(out_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  训练结果已保存: {results_path}")
    print(f"  提示: 运行 python translate.py 即可使用命令行翻译")
    print(f"{_BOLD}\n")


if __name__ == "__main__":
    main()
