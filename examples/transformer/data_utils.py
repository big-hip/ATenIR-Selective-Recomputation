"""
data_utils.py

Multi30k EN→DE 翻译数据集的下载、BPE 分词、Dataset 和 DataLoader 构建。
零额外依赖（使用已安装的 tokenizers 库进行 BPE 分词）。

用法::

    from data_utils import build_dataloaders

    train_loader, val_loader, test_loader, src_tok, tgt_tok = build_dataloaders(
        vocab_size=5000, max_len=64, batch_size=64,
    )
"""
import os
import random
import ssl
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

# ═══════════════════════════════════════════════════════════════════════════════
#  常量
# ═══════════════════════════════════════════════════════════════════════════════

_DATA_DIR = Path(__file__).parent / "data" / "multi30k"

_URLS = {
    "train": "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz",
    "val": "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz",
    "test": "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz",
}

# 解压后的文件名映射
_EXTRACTED = {
    "train_en": "train.en",
    "train_de": "train.de",
    "val_en": "val.en",
    "val_de": "val.de",
    "test_en": "test.en",
    "test_de": "test.de",
}

# 特殊 token
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


# ═══════════════════════════════════════════════════════════════════════════════
#  数据下载
# ═══════════════════════════════════════════════════════════════════════════════

def _download_and_extract(url: str, dest_dir: Path) -> None:
    """下载 tar.gz 并解压到 dest_dir。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dest_dir / url.split("/")[-1]

    if not tar_path.exists():
        print(f"  下载 {url} ...")
        try:
            urllib.request.urlretrieve(url, tar_path)
        except (urllib.error.URLError, ssl.SSLError) as e:
            print(f"  [错误] 下载失败: {e}")
            print(f"  请手动下载以下文件并放到 {dest_dir}/:")
            for u in _URLS.values():
                print(f"    {u}")
            raise

    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(path=dest_dir)


def download_multi30k(data_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    下载 Multi30k 数据集到 data_dir。如果文件已存在则跳过。

    Returns
    -------
    dict : 文件名标识 → 文件路径
    """
    data_dir = data_dir or _DATA_DIR

    # 检查是否所有文件都已存在
    paths = {k: data_dir / v for k, v in _EXTRACTED.items()}
    if all(p.exists() for p in paths.values()):
        print(f"  Multi30k 数据已存在于 {data_dir}")
        return paths

    print(f"  下载 Multi30k 到 {data_dir} ...")
    for split_name, url in _URLS.items():
        _download_and_extract(url, data_dir)

    # 验证文件
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"解压后未找到预期文件: {p}")

    return paths


# ═══════════════════════════════════════════════════════════════════════════════
#  BPE 分词器
# ═══════════════════════════════════════════════════════════════════════════════

def build_tokenizer(
    corpus: List[str],
    vocab_size: int,
    save_path: Path,
) -> Tokenizer:
    """
    训练 BPE 分词器或从缓存加载。

    Parameters
    ----------
    corpus : 训练语料（每行一个句子）
    vocab_size : 目标词表大小
    save_path : 分词器 JSON 保存路径

    Returns
    -------
    Tokenizer : 训练好的分词器
    """
    if save_path.exists():
        print(f"  加载已有分词器: {save_path}")
        tok = Tokenizer.from_file(str(save_path))
        return tok

    print(f"  训练 BPE 分词器 (vocab_size={vocab_size}) ...")
    tok = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tok.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
    )
    tok.train_from_iterator(corpus, trainer=trainer)

    # 自动在编码结果前后插入 [BOS] 和 [EOS]
    tok.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        special_tokens=[
            (BOS_TOKEN, tok.token_to_id(BOS_TOKEN)),
            (EOS_TOKEN, tok.token_to_id(EOS_TOKEN)),
        ],
    )

    # 启用 padding
    tok.enable_padding(pad_id=PAD_ID, pad_token=PAD_TOKEN)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(save_path))
    print(f"  分词器已保存: {save_path} (vocab={tok.get_vocab_size()})")
    return tok


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset & DataLoader
# ═══════════════════════════════════════════════════════════════════════════════

class TranslationDataset(Dataset):
    """Multi30k 翻译 Dataset：存储分词后的 token ID 对。"""

    def __init__(
        self,
        src_lines: List[str],
        tgt_lines: List[str],
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        max_len: int = 64,
    ):
        assert len(src_lines) == len(tgt_lines)
        self.src_ids = []
        self.tgt_ids = []

        src_tokenizer.enable_truncation(max_length=max_len)
        tgt_tokenizer.enable_truncation(max_length=max_len)

        for src_line, tgt_line in zip(src_lines, tgt_lines):
            src_enc = src_tokenizer.encode(src_line.strip())
            tgt_enc = tgt_tokenizer.encode(tgt_line.strip())
            self.src_ids.append(src_enc.ids)
            self.tgt_ids.append(tgt_enc.ids)

        # 重置 truncation 避免影响后续使用
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
    """返回固定长度的 collate_fn，所有 batch pad 到 max_len，避免 torch.compile 因 shape 变化重编译。"""
    def collate_fn(batch):
        src_batch, tgt_batch = zip(*batch)
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_ID)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_ID)
        # 统一 pad 到 max_len，消除 shape 变化
        B = src_padded.size(0)
        if src_padded.size(1) < max_len:
            src_padded = torch.nn.functional.pad(src_padded, (0, max_len - src_padded.size(1)), value=PAD_ID)
        if tgt_padded.size(1) < max_len:
            tgt_padded = torch.nn.functional.pad(tgt_padded, (0, max_len - tgt_padded.size(1)), value=PAD_ID)
        return src_padded, tgt_padded
    return collate_fn


def _read_lines(path: Path) -> List[str]:
    """读取文本文件为行列表。"""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ═══════════════════════════════════════════════════════════════════════════════
#  合成数据 fallback（网络不可用时使用）
# ═══════════════════════════════════════════════════════════════════════════════

# 50 个英语词 + 50 个对应的"伪德语"词，构成确定性映射
_EN_WORDS = [
    "the", "a", "man", "woman", "child", "dog", "cat", "bird", "car", "house",
    "is", "are", "was", "runs", "walks", "sits", "eats", "drinks", "plays", "reads",
    "big", "small", "red", "blue", "green", "old", "new", "fast", "slow", "happy",
    "in", "on", "at", "by", "with", "from", "to", "near", "under", "over",
    "and", "but", "or", "so", "then", "now", "here", "there", "very", "not",
]
_DE_WORDS = [
    "der", "ein", "mann", "frau", "kind", "hund", "katze", "vogel", "auto", "haus",
    "ist", "sind", "war", "rennt", "geht", "sitzt", "isst", "trinkt", "spielt", "liest",
    "gross", "klein", "rot", "blau", "gruen", "alt", "neu", "schnell", "langsam", "gluecklich",
    "in", "auf", "an", "bei", "mit", "von", "zu", "nahe", "unter", "ueber",
    "und", "aber", "oder", "also", "dann", "jetzt", "hier", "dort", "sehr", "nicht",
]
_EN2DE = dict(zip(_EN_WORDS, _DE_WORDS))


def _generate_synthetic_data(
    n_train: int = 10000,
    n_val: int = 1000,
    n_test: int = 1000,
    min_len: int = 5,
    max_len: int = 15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    生成合成翻译数据：每个英语句子由随机词组成，
    对应的德语句子是逐词映射（确定性"翻译"）。
    模型可以学到这种映射，从而验证训练正确性。
    """
    rng = random.Random(seed)

    def _make_pair():
        length = rng.randint(min_len, max_len)
        en_words = [rng.choice(_EN_WORDS) for _ in range(length)]
        de_words = [_EN2DE[w] for w in en_words]
        return " ".join(en_words), " ".join(de_words)

    def _make_split(n):
        en_lines, de_lines = [], []
        for _ in range(n):
            en, de = _make_pair()
            en_lines.append(en)
            de_lines.append(de)
        return en_lines, de_lines

    train_en, train_de = _make_split(n_train)
    val_en, val_de = _make_split(n_val)
    test_en, test_de = _make_split(n_test)

    return train_en, train_de, val_en, val_de, test_en, test_de


def _multi30k_available(data_dir: Path) -> bool:
    """检查 Multi30k 文件是否已存在。"""
    paths = {k: data_dir / v for k, v in _EXTRACTED.items()}
    return all(p.exists() for p in paths.values())


def build_dataloaders(
    vocab_size: int = 5000,
    max_len: int = 64,
    batch_size: int = 64,
    data_dir: Optional[Path] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Tokenizer, Tokenizer]:
    """
    端到端构建数据管线。优先使用 Multi30k 真实数据；
    若数据不存在且下载失败，则回退到合成数据。

    Returns
    -------
    (train_loader, val_loader, test_loader, src_tokenizer, tgt_tokenizer)
    """
    data_dir = data_dir or _DATA_DIR

    # 尝试加载 Multi30k 真实数据
    use_synthetic = False
    if _multi30k_available(data_dir):
        paths = {k: data_dir / v for k, v in _EXTRACTED.items()}
        print(f"  Multi30k 数据已存在于 {data_dir}")
        train_en = _read_lines(paths["train_en"])
        train_de = _read_lines(paths["train_de"])
        val_en = _read_lines(paths["val_en"])
        val_de = _read_lines(paths["val_de"])
        test_en = _read_lines(paths["test_en"])
        test_de = _read_lines(paths["test_de"])
    else:
        # 尝试下载
        try:
            paths = download_multi30k(data_dir)
            train_en = _read_lines(paths["train_en"])
            train_de = _read_lines(paths["train_de"])
            val_en = _read_lines(paths["val_en"])
            val_de = _read_lines(paths["val_de"])
            test_en = _read_lines(paths["test_en"])
            test_de = _read_lines(paths["test_de"])
        except Exception as e:
            print(f"\n  [回退] Multi30k 下载失败 ({type(e).__name__})")
            print("  使用合成翻译数据（确定性词级映射）验证训练正确性 ...")
            use_synthetic = True
            train_en, train_de, val_en, val_de, test_en, test_de = _generate_synthetic_data()

    dataset_name = "合成数据" if use_synthetic else "Multi30k"
    print(f"  数据集: {dataset_name}")
    print(f"  数据规模: train={len(train_en)}, val={len(val_en)}, test={len(test_en)}")

    # 分词器（合成数据用较小 vocab）
    tok_vocab = min(vocab_size, 500) if use_synthetic else vocab_size
    tok_dir = data_dir / ("synthetic" if use_synthetic else "")
    tok_dir.mkdir(parents=True, exist_ok=True)

    src_tok = build_tokenizer(train_en, tok_vocab, tok_dir / "tokenizer_en.json")
    tgt_tok = build_tokenizer(train_de, tok_vocab, tok_dir / "tokenizer_de.json")

    # 构建 Dataset
    train_ds = TranslationDataset(train_en, train_de, src_tok, tgt_tok, max_len)
    val_ds = TranslationDataset(val_en, val_de, src_tok, tgt_tok, max_len)
    test_ds = TranslationDataset(test_en, test_de, src_tok, tgt_tok, max_len)

    # 构建 DataLoader（统一 pad 到 max_len + drop_last，避免 torch.compile 重编译）
    _collate = make_collate_fn(max_len)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=_collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_collate, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_collate, drop_last=False,
    )

    print(f"  DataLoader 构建完成: "
          f"train={len(train_loader)} batches, "
          f"val={len(val_loader)} batches, "
          f"test={len(test_loader)} batches")

    return train_loader, val_loader, test_loader, src_tok, tgt_tok
