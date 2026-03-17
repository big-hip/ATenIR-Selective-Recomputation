"""
zh_en_train.py

中文→英文句子翻译训练脚本。
支持两种训练模式：
1. 纯 eager 训练
2. torch.compile + ATenIR 选择性重计算训练

默认优先读取用户提供的中英平行语料；如果未提供语料，则自动下载公开的
Mandarin Chinese-English 平行语料用于训练。

用法::

    cd examples/transformer
    python zh_en_train.py --train-zh path/to/train.zh --train-en path/to/train.en
    python zh_en_train.py --train-zh path/to/train.zh --train-en path/to/train.en \
        --val-zh path/to/val.zh --val-en path/to/val.en
    python zh_en_train.py --eager
    python zh_en_train.py --strategy '{"6": 0}'
"""
import argparse
import json
import math
import os
import random
import ssl
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import Transformer
from aten_recompute.core import CompilerBackend, inject_layer_tags

# ═══════════════════════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════════════════════

D_MODEL = 256
NUM_HEADS = 4
NUM_LAYERS = 4
D_FF = 1024
MAX_SEQ_LENGTH = 48
DROPOUT = 0.10

BATCH_SIZE = 64
NUM_EPOCHS = 30
LR = 6e-4
WARMUP_STEPS = 300
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 1.0
SEED = 42
VOCAB_SIZE = 5000

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = Path(__file__).parent
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
TOKENIZER_DIR = SCRIPT_DIR / "data" / "zh_en"
DATA_DIR = SCRIPT_DIR / "data" / "zh_en"
DEFAULT_DATASET_URL = "https://www.manythings.org/anki/cmn-eng.zip"
DEFAULT_DATASET_ZIP = DATA_DIR / "cmn-eng.zip"
DEFAULT_DATASET_TSV = DATA_DIR / "cmn.txt"

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


def seed_everything(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = torch.cuda.is_available()


def init_weights(model: nn.Module):
    for name, parameter in model.named_parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)
        elif "bias" in name:
            nn.init.zeros_(parameter)


def build_lr_lambda(warmup_steps: int):
    def lr_lambda(step):
        step = max(step, 1)
        warmup_factor = min(step / warmup_steps, 1.0)
        decay_factor = (warmup_steps / max(step, warmup_steps)) ** 0.5
        return warmup_factor * decay_factor
    return lr_lambda


def build_model(src_vocab_size: int, tgt_vocab_size: int) -> Transformer:
    return Transformer(
        src_vocab_size,
        tgt_vocab_size,
        D_MODEL,
        NUM_HEADS,
        NUM_LAYERS,
        D_FF,
        MAX_SEQ_LENGTH,
        DROPOUT,
        padding_idx=PAD_ID,
    )


def save_checkpoint(model, tag: str, src_vocab_size: int, tgt_vocab_size: int,
                    optimizer=None, scheduler=None, epoch: int = None, best_val_loss: float = None):
    """保存模型与可选训练状态（optimizer/scheduler/epoch）。"""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
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
            "source_language": "zh",
            "target_language": "en",
        },
    }
    if optimizer is not None:
        try:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        except Exception:
            pass
    if scheduler is not None:
        try:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        except Exception:
            pass
    if epoch is not None:
        checkpoint["epoch"] = int(epoch)
    if best_val_loss is not None:
        try:
            checkpoint["best_val_loss"] = float(best_val_loss)
        except Exception:
            pass

    path = CHECKPOINT_DIR / f"transformer_zh_en_{tag}.pt"
    torch.save(checkpoint, path)
    print(f"  模型已保存: {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)")


def _read_lines(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def download_default_corpus() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DEFAULT_DATASET_TSV.exists():
        print(f"  已找到公开语料: {DEFAULT_DATASET_TSV}")
        return DEFAULT_DATASET_TSV

    if not DEFAULT_DATASET_ZIP.exists():
        print(f"  下载公开中英平行语料: {DEFAULT_DATASET_URL}")
        try:
            urllib.request.urlretrieve(DEFAULT_DATASET_URL, DEFAULT_DATASET_ZIP)
        except (urllib.error.URLError, ssl.SSLError) as exc:
            raise RuntimeError(
                "公开中文-英文语料下载失败，请检查网络，或手动提供 --train-zh / --train-en。"
            ) from exc

    with zipfile.ZipFile(DEFAULT_DATASET_ZIP, "r") as zip_file:
        zip_file.extractall(DATA_DIR)

    if not DEFAULT_DATASET_TSV.exists():
        raise FileNotFoundError(f"下载完成后未找到语料文件: {DEFAULT_DATASET_TSV}")

    return DEFAULT_DATASET_TSV


def load_default_parallel_corpus(limit: int = 0) -> Tuple[List[str], List[str]]:
    dataset_path = download_default_corpus()
    zh_lines: List[str] = []
    en_lines: List[str] = []
    seen_pairs = set()

    with open(dataset_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            en_text, zh_text = parts[0].strip(), parts[1].strip()
            if not en_text or not zh_text:
                continue
            pair = (zh_text, en_text)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            zh_lines.append(zh_text)
            en_lines.append(en_text)
            if limit > 0 and len(zh_lines) >= limit:
                break

    if not zh_lines:
        raise RuntimeError("公开中文-英文语料为空，无法训练。")
    return zh_lines, en_lines


def _normalize_zh_text(text: str) -> str:
    return " ".join(list(text.strip().replace(" ", "")))


def _normalize_en_text(text: str) -> str:
    return " ".join(text.strip().split())


def _filter_parallel_pairs(
    src_lines: Sequence[str],
    tgt_lines: Sequence[str],
    max_len: int,
) -> Tuple[List[str], List[str]]:
    kept_src: List[str] = []
    kept_tgt: List[str] = []

    max_src_len = max_len - 2
    max_tgt_len = max_len - 2

    for src_text, tgt_text in zip(src_lines, tgt_lines):
        src_len = len(src_text.split())
        tgt_len = len(tgt_text.split())
        if src_len == 0 or tgt_len == 0:
            continue
        if src_len > max_src_len or tgt_len > max_tgt_len:
            continue
        kept_src.append(src_text)
        kept_tgt.append(tgt_text)

    if not kept_src:
        raise RuntimeError("过滤语料后没有可用样本，请增大 --max-len 或检查输入数据。")
    return kept_src, kept_tgt


def _split_parallel_data(
    zh_lines: Sequence[str],
    en_lines: Sequence[str],
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    pairs = list(zip(zh_lines, en_lines))
    rng = random.Random(seed)
    rng.shuffle(pairs)

    total = len(pairs)
    train_end = max(1, int(total * train_ratio))
    val_end = max(train_end + 1, int(total * (train_ratio + val_ratio)))
    val_end = min(val_end, total)

    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    if not val_pairs:
        val_pairs = train_pairs[-1:]
    if not test_pairs:
        test_pairs = val_pairs[-1:]

    def unzip(data_pairs):
        src, tgt = zip(*data_pairs)
        return list(src), list(tgt)

    train_zh, train_en = unzip(train_pairs)
    val_zh, val_en = unzip(val_pairs)
    test_zh, test_en = unzip(test_pairs)
    return train_zh, train_en, val_zh, val_en, test_zh, test_en


def load_parallel_corpus(args) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], str]:
    if args.train_zh and args.train_en:
        train_zh = _read_lines(Path(args.train_zh))
        train_en = _read_lines(Path(args.train_en))
        if len(train_zh) != len(train_en):
            raise ValueError("训练集中文与英文行数不一致。")

        if args.val_zh and args.val_en and args.test_zh and args.test_en:
            val_zh = _read_lines(Path(args.val_zh))
            val_en = _read_lines(Path(args.val_en))
            test_zh = _read_lines(Path(args.test_zh))
            test_en = _read_lines(Path(args.test_en))
            if len(val_zh) != len(val_en):
                raise ValueError("验证集中文与英文行数不一致。")
            if len(test_zh) != len(test_en):
                raise ValueError("测试集中文与英文行数不一致。")
            dataset_name = "用户提供语料"
        else:
            train_zh, train_en, val_zh, val_en, test_zh, test_en = _split_parallel_data(
                train_zh, train_en, seed=args.seed,
            )
            dataset_name = "用户提供语料（自动切分）"
    else:
        zh_lines, en_lines = load_default_parallel_corpus(limit=args.max_samples)
        train_zh, train_en, val_zh, val_en, test_zh, test_en = _split_parallel_data(
            zh_lines, en_lines, seed=args.seed,
        )
        dataset_name = "ManyThings/Tatoeba Mandarin-English"

    return train_zh, train_en, val_zh, val_en, test_zh, test_en, dataset_name


def build_tokenizer(corpus: List[str], vocab_size: int, save_path: Path) -> Tokenizer:
    if save_path.exists() and not os.getenv("ZH_EN_REBUILD_TOKENIZER"):
        print(f"  加载已有分词器: {save_path}")
        return Tokenizer.from_file(str(save_path))

    print(f"  训练分词器: {save_path.name} (vocab_size={vocab_size})")
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=1,
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        special_tokens=[
            (BOS_TOKEN, tokenizer.token_to_id(BOS_TOKEN)),
            (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
        ],
    )
    tokenizer.enable_padding(pad_id=PAD_ID, pad_token=PAD_TOKEN)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path))
    return tokenizer


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_lines: List[str],
        tgt_lines: List[str],
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        max_len: int,
    ):
        if len(src_lines) != len(tgt_lines):
            raise ValueError("源语言与目标语言样本数量不一致。")

        self.src_ids = []
        self.tgt_ids = []

        src_tokenizer.enable_truncation(max_length=max_len)
        tgt_tokenizer.enable_truncation(max_length=max_len)

        for src_line, tgt_line in zip(src_lines, tgt_lines):
            src_ids = src_tokenizer.encode(src_line).ids
            tgt_ids = tgt_tokenizer.encode(tgt_line).ids
            self.src_ids.append(src_ids)
            self.tgt_ids.append(tgt_ids)

        src_tokenizer.no_truncation()
        tgt_tokenizer.no_truncation()

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.src_ids[idx], dtype=torch.long),
            torch.tensor(self.tgt_ids[idx], dtype=torch.long),
        )


def make_collate_fn(max_len: int):
    def collate_fn(batch):
        src_batch, tgt_batch = zip(*batch)
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_ID)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_ID)
        if src_padded.size(1) < max_len:
            src_padded = torch.nn.functional.pad(
                src_padded,
                (0, max_len - src_padded.size(1)),
                value=PAD_ID,
            )
        if tgt_padded.size(1) < max_len:
            tgt_padded = torch.nn.functional.pad(
                tgt_padded,
                (0, max_len - tgt_padded.size(1)),
                value=PAD_ID,
            )
        return src_padded, tgt_padded
    return collate_fn


def build_dataloaders(args):
    train_zh, train_en, val_zh, val_en, test_zh, test_en, dataset_name = load_parallel_corpus(args)

    train_zh = [_normalize_zh_text(text) for text in train_zh]
    val_zh = [_normalize_zh_text(text) for text in val_zh]
    test_zh = [_normalize_zh_text(text) for text in test_zh]
    train_en = [_normalize_en_text(text) for text in train_en]
    val_en = [_normalize_en_text(text) for text in val_en]
    test_en = [_normalize_en_text(text) for text in test_en]

    train_zh, train_en = _filter_parallel_pairs(train_zh, train_en, args.max_len)
    val_zh, val_en = _filter_parallel_pairs(val_zh, val_en, args.max_len)
    test_zh, test_en = _filter_parallel_pairs(test_zh, test_en, args.max_len)

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    src_tok = build_tokenizer(train_zh, args.vocab_size, TOKENIZER_DIR / "tokenizer_zh.json")
    tgt_tok = build_tokenizer(train_en, args.vocab_size, TOKENIZER_DIR / "tokenizer_en.json")

    train_ds = TranslationDataset(train_zh, train_en, src_tok, tgt_tok, args.max_len)
    val_ds = TranslationDataset(val_zh, val_en, src_tok, tgt_tok, args.max_len)
    test_ds = TranslationDataset(test_zh, test_en, src_tok, tgt_tok, args.max_len)

    collate_fn = make_collate_fn(args.max_len)
    use_cuda = DEVICE.type == "cuda"
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": use_cuda,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=len(train_ds) >= args.batch_size,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader, src_tok, tgt_tok, dataset_name


def train_one_epoch(model, loader, criterion, optimizer, scaler, amp_enabled, scheduler=None):
    model.train()
    total_loss, total_tokens = 0.0, 0
    for src, tgt in loader:
        src = src.to(DEVICE, non_blocking=True)
        tgt = tgt.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=DEVICE.type, enabled=amp_enabled):
            output = model(src, tgt[:, :-1])
            loss = criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt[:, 1:].contiguous().view(-1),
            )

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
        non_pad_tokens = (tgt[:, 1:] != PAD_ID).sum().item()
        total_loss += loss.item() * non_pad_tokens
        total_tokens += non_pad_tokens
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, amp_enabled):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for src, tgt in loader:
        src = src.to(DEVICE, non_blocking=True)
        tgt = tgt.to(DEVICE, non_blocking=True)
        with torch.amp.autocast(device_type=DEVICE.type, enabled=amp_enabled):
            output = model(src, tgt[:, :-1])
            loss = criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt[:, 1:].contiguous().view(-1),
            )
        non_pad_tokens = (tgt[:, 1:] != PAD_ID).sum().item()
        total_loss += loss.item() * non_pad_tokens
        total_tokens += non_pad_tokens
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def greedy_decode(model, src, max_len):
    model.eval()
    batch_size = src.size(0)

    src_padding_mask = torch.eq(src, model.padding_idx)
    src_embedded = model.dropout1(model.positional_encoding1(model.encoder_embedding(src)))
    enc_output = src_embedded
    for encoder_layer in model.encoder_layers:
        enc_output = encoder_layer(enc_output, src_padding_mask)

    tgt_ids = torch.full((batch_size, 1), BOS_ID, dtype=torch.long, device=src.device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

    for _ in range(max_len - 1):
        tgt_padding_mask = torch.eq(tgt_ids, model.padding_idx)
        tgt_len = tgt_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=src.device),
            diagonal=1,
        ).bool()

        tgt_embedded = model.dropout2(
            model.positional_encoding2(model.decoder_embedding(tgt_ids))
        )
        dec_output = tgt_embedded
        for decoder_layer in model.decoder_layers:
            dec_output = decoder_layer(
                dec_output,
                enc_output,
                causal_mask,
                src_padding_mask,
                tgt_padding_mask,
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


def compute_corpus_bleu(references: List[str], hypotheses: List[str], max_n: int = 4) -> float:
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
            clipped = {ng: min(count, ref_ngrams.get(ng, 0)) for ng, count in hyp_ngrams.items()}
            clipped_counts[n - 1] += sum(clipped.values())
            total_counts[n - 1] += max(len(hyp_tokens) - n + 1, 0)

    log_precisions = 0.0
    for n in range(max_n):
        if total_counts[n] == 0 or clipped_counts[n] == 0:
            return 0.0
        log_precisions += math.log(clipped_counts[n] / total_counts[n])
    log_precisions /= max_n
    brevity_penalty = min(1.0 - ref_len / hyp_len, 0.0) if hyp_len > 0 else -1e9
    return 100.0 * math.exp(brevity_penalty + log_precisions)


@torch.no_grad()
def compute_test_bleu(model, test_loader, tgt_tokenizer, max_len: int) -> float:
    references, hypotheses = [], []
    for src, tgt in test_loader:
        src = src.to(DEVICE)
        pred_ids = greedy_decode(model, src, max_len)
        for batch_index in range(src.size(0)):
            pred = pred_ids[batch_index].tolist()
            if EOS_ID in pred:
                pred = pred[:pred.index(EOS_ID)]
            pred = [token for token in pred if token not in (BOS_ID, PAD_ID)]
            hypothesis = _normalize_en_text(tgt_tokenizer.decode(pred))

            ref = tgt[batch_index].tolist()
            ref = [token for token in ref if token not in (BOS_ID, EOS_ID, PAD_ID)]
            reference = _normalize_en_text(tgt_tokenizer.decode(ref))

            references.append(reference)
            hypotheses.append(hypothesis)
    return compute_corpus_bleu(references, hypotheses)


@torch.no_grad()
def run_demo_translation(model, src_tokenizer, tgt_tokenizer, text: str, max_len: int):
    model.eval()
    normalized_text = _normalize_zh_text(text)
    src_ids = src_tokenizer.encode(normalized_text).ids[:max_len]
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)
    if src_tensor.size(1) < max_len:
        src_tensor = torch.nn.functional.pad(
            src_tensor,
            (0, max_len - src_tensor.size(1)),
            value=PAD_ID,
        )
    pred_ids = greedy_decode(model, src_tensor, max_len)[0].tolist()
    if EOS_ID in pred_ids:
        pred_ids = pred_ids[:pred_ids.index(EOS_ID)]
    pred_ids = [token for token in pred_ids if token not in (BOS_ID, PAD_ID)]
    return tgt_tokenizer.decode(pred_ids)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-zh", type=str, help="训练集中文文本路径，每行一句")
    parser.add_argument("--train-en", type=str, help="训练集英文文本路径，每行一句")
    parser.add_argument("--val-zh", type=str, help="验证集中文文本路径")
    parser.add_argument("--val-en", type=str, help="验证集英文文本路径")
    parser.add_argument("--test-zh", type=str, help="测试集中文文本路径")
    parser.add_argument("--test-en", type=str, help="测试集英文文本路径")
    parser.add_argument("--eager", action="store_true", help="纯 eager 训练，不使用 torch.compile")
    parser.add_argument("--strategy", type=str, default=None, help="重计算策略 JSON，例如 '{\"6\": 0}'")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-len", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--max-samples", type=int, default=0, help="限制自动下载语料的样本数，0 表示使用全部样本")
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 0), help="DataLoader worker 数")
    parser.add_argument("--disable-amp", action="store_true", help="禁用 CUDA AMP 混合精度")
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--demo-text", type=str, default="今天天气很好。", help="训练完成后做一次单句翻译演示")
    parser.add_argument("--resume", action="store_true", help="从 checkpoints/transformer_zh_en_best.pt 接着训练（若存在）")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="指定要从哪个 checkpoint 路径恢复训练")
    return parser.parse_args()


def main():
    args = parse_args()
    if bool(args.train_zh) != bool(args.train_en):
        raise ValueError("必须同时提供 --train-zh 和 --train-en。")
    if bool(args.val_zh) != bool(args.val_en):
        raise ValueError("必须同时提供 --val-zh 和 --val-en。")
    if bool(args.test_zh) != bool(args.test_en):
        raise ValueError("必须同时提供 --test-zh 和 --test-en。")

    os.environ.setdefault("RECOMPUTE_LOG_LEVEL", "WARNING")
    torch._dynamo.config.cache_size_limit = 64
    torch.set_float32_matmul_precision("high")
    seed_everything(args.seed)
    amp_enabled = DEVICE.type == "cuda" and not args.disable_amp

    strategy_config = json.loads(args.strategy or os.getenv("RECOMPUTE", '{"6": 0}'))
    strategy_key = next(iter(strategy_config), "0")
    use_eager = args.eager

    mode_str = "Eager (无编译)" if use_eager else f"ATenIR 编译 (策略 {strategy_key})"
    print(f"\n{_BOLD}")
    print("  中文→英文 Transformer 训练")
    print(_BOLD)
    print(f"  设备:       {DEVICE}")
    print(f"  训练模式:   {mode_str}")
    if not use_eager:
        print(f"  重计算策略: {_strategy_desc(strategy_config)}")
    print(f"  模型:       {NUM_LAYERS}L-{D_MODEL}d-{NUM_HEADS}h-{D_FF}ff")
    print(f"  训练:       {args.epochs} epochs, batch={args.batch_size}, lr={LR}")
    print(f"  AMP:        {'on' if amp_enabled else 'off'}")
    print(f"  DataLoader: workers={args.num_workers}, pin_memory={'on' if DEVICE.type == 'cuda' else 'off'}")
    print(_LINE)

    print("\n[阶段 1/4] 构建数据管线")
    print(_LINE)
    train_loader, val_loader, test_loader, src_tok, tgt_tok, dataset_name = build_dataloaders(args)
    src_vocab_size = src_tok.get_vocab_size()
    tgt_vocab_size = tgt_tok.get_vocab_size()
    print(f"  数据集:     {dataset_name}")
    print(f"  src_vocab={src_vocab_size}, tgt_vocab={tgt_vocab_size}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=LABEL_SMOOTHING)

    print("\n[阶段 2/4] 构建模型")
    print(_LINE)
    model = build_model(src_vocab_size, tgt_vocab_size).to(DEVICE)
    init_weights(model)
    param_count = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    print(f"  模型参数量: {param_count:.1f}M")

    if use_eager:
        train_model = model
        print("  模式: eager (无 torch.compile)")
    else:
        encoder_layers = [(layer, idx) for idx, layer in enumerate(model.encoder_layers)]
        decoder_layers = [
            (layer, len(model.encoder_layers) + idx)
            for idx, layer in enumerate(model.decoder_layers)
        ]
        inject_layer_tags(encoder_layers + decoder_layers)
        backend = CompilerBackend(strategy_config=strategy_config, save_ir=False)
        train_model = torch.compile(model, backend=backend, dynamic=False)
        print("  模式: torch.compile + ATenIR")
        print("  (首次前向会触发编译，耗时会明显更长)")

    eval_model = model

    print("\n[阶段 3/4] 训练")
    print(_LINE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        build_lr_lambda(WARMUP_STEPS),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    start_time = time.time()

    # 恢复训练（如果用户请求）
    start_epoch = 0
    resume_path = None
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
    elif args.resume:
        resume_path = CHECKPOINT_DIR / "transformer_zh_en_best.pt"

    if resume_path is not None and resume_path.exists():
        print(f"  从 checkpoint 恢复: {resume_path}")
        ckpt = torch.load(resume_path, map_location=DEVICE)
        try:
            model.load_state_dict(ckpt["model_state_dict"])
            print("  模型参数已加载。")
        except Exception as exc:
            print(f"  警告: 无法完整加载模型权重: {exc}")
        # optimizer/scheduler 需要在创建后恢复
        start_epoch = int(ckpt.get("epoch", 0))
        if "best_val_loss" in ckpt:
            try:
                best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
            except Exception:
                pass
        # 尝试恢复 optimizer / scheduler 状态（若 checkpoint 中包含）
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                print("  optimizer 状态已恢复。")
            except Exception as exc:
                print(f"  警告: 无法恢复 optimizer 状态: {exc}")
        if "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                print("  scheduler 状态已恢复。")
            except Exception as exc:
                print(f"  警告: 无法恢复 scheduler 状态: {exc}")

    # epoch 迭代范围：若从 checkpoint 恢复，则将 args.epochs 视为继续训练的轮数
    if start_epoch > 0:
        total_epochs = start_epoch + int(args.epochs)
        epoch_iter = range(start_epoch, total_epochs)
    else:
        epoch_iter = range(int(args.epochs))

    # 若 checkpoint 中包含 optimizer/scheduler 的状态，则在创建后恢复（下一步）

    for epoch in epoch_iter:
        train_loss = train_one_epoch(
            train_model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            amp_enabled,
            scheduler=scheduler,
        )
        val_loss = evaluate(eval_model, val_loader, criterion, amp_enabled)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, "best", src_vocab_size, tgt_vocab_size,
                            optimizer=optimizer, scheduler=scheduler, epoch=epoch + 1,
                            best_val_loss=best_val_loss)
            improved = " *"

        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - start_time
        print(
            f"  Epoch {epoch + 1:2d}/{args.epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"lr={current_lr:.2e} | time={elapsed:.1f}s{improved}"
        )

    total_time = time.time() - start_time

    print("\n[阶段 4/4] 测试评估")
    print(_LINE)
    bleu = compute_test_bleu(model, test_loader, tgt_tok, args.max_len)
    print(f"  Test BLEU-4:    {bleu:.2f}")
    print(f"  Best val_loss:  {best_val_loss:.4f}")
    print(f"  Total time:     {total_time:.1f}s")

    tag = "eager" if use_eager else f"strat{strategy_key}"
    save_checkpoint(model, tag, src_vocab_size, tgt_vocab_size,
                    optimizer=optimizer, scheduler=scheduler, epoch=epoch + 1,
                    best_val_loss=best_val_loss)

    from aten_recompute.utils.save_ir import _default_ir_dir

    model_name = os.getenv("MODEL_NAME", "Transformer")
    out_dir = _default_ir_dir(model_name, subfolder="custom_training")
    results = {
        "task": "zh_to_en_translation",
        "dataset": dataset_name,
        "mode": "eager" if use_eager else "compiled",
        "strategy": strategy_config if not use_eager else None,
        "strategy_desc": _strategy_desc(strategy_config) if not use_eager else "eager",
        "model_config": {
            "d_model": D_MODEL,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "d_ff": D_FF,
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "source_language": "zh",
            "target_language": "en",
        },
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": LR,
            "warmup_steps": WARMUP_STEPS,
            "label_smoothing": LABEL_SMOOTHING,
            "grad_clip": GRAD_CLIP,
            "seed": args.seed,
            "max_len": args.max_len,
            "vocab_size": args.vocab_size,
        },
        "results": {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "test_bleu": bleu,
            "total_time_s": round(total_time, 1),
        },
    }
    results_path = os.path.join(out_dir, f"zh_en_train_{tag}.json")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    demo_translation = run_demo_translation(model, src_tok, tgt_tok, args.demo_text, args.max_len)

    print(f"\n{_BOLD}")
    print("  完成。结果已保存:")
    print(f"    训练结果: {results_path}")
    print(f"    模型:     checkpoints/transformer_zh_en_{tag}.pt")
    print("    分词器:   data/zh_en/tokenizer_zh.json, data/zh_en/tokenizer_en.json")
    print(f"    演示输入: {args.demo_text}")
    print(f"    演示输出: {demo_translation}")
    print(f"{_BOLD}\n")


if __name__ == "__main__":
    main()