"""
translate.py

交互式命令行翻译工具。加载训练好的 Transformer 模型，实时 EN→DE 翻译。

用法::

    cd examples/transformer
    python translate.py                          # 默认加载 best 模型
    python translate.py --checkpoint checkpoints/transformer_eager.pt
    python translate.py --text "A dog runs in the park"
    # 或通过 run.sh:
    ./run.sh translate
"""
import argparse
import os
import sys
from pathlib import Path

import torch

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import Transformer
from data_utils import BOS_ID, EOS_ID, PAD_ID

from tokenizers import Tokenizer

# ═══════════════════════════════════════════════════════════════════════════════
#  路径常量
# ═══════════════════════════════════════════════════════════════════════════════

_SCRIPT_DIR = Path(__file__).parent
_CHECKPOINT_DIR = _SCRIPT_DIR / "checkpoints"
_DATA_DIR = _SCRIPT_DIR / "data" / "multi30k"

# ═══════════════════════════════════════════════════════════════════════════════
#  模型加载
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: Path, device: torch.device):
    """从 checkpoint 加载 Transformer 模型和配置。"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]

    model = Transformer(
        cfg["src_vocab_size"], cfg["tgt_vocab_size"],
        cfg["d_model"], cfg["num_heads"], cfg["num_layers"],
        cfg["d_ff"], cfg["max_seq_length"], cfg["dropout"],
        padding_idx=cfg["padding_idx"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, cfg


def load_tokenizers():
    """加载已训练的 BPE 分词器。"""
    src_path = _DATA_DIR / "tokenizer_en.json"
    tgt_path = _DATA_DIR / "tokenizer_de.json"

    if not src_path.exists() or not tgt_path.exists():
        # 尝试 synthetic 路径
        src_path = _DATA_DIR / "synthetic" / "tokenizer_en.json"
        tgt_path = _DATA_DIR / "synthetic" / "tokenizer_de.json"

    if not src_path.exists() or not tgt_path.exists():
        print("错误: 找不到分词器文件。请先运行 train_multi30k.py 训练模型。")
        print(f"  期望路径: {_DATA_DIR / 'tokenizer_en.json'}")
        sys.exit(1)

    src_tok = Tokenizer.from_file(str(src_path))
    tgt_tok = Tokenizer.from_file(str(tgt_path))
    return src_tok, tgt_tok


# ═══════════════════════════════════════════════════════════════════════════════
#  贪心解码
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def translate_sentence(model, src_text: str, src_tok, tgt_tok, device, max_len: int = 64):
    """
    翻译一句英语文本为德语。

    Parameters
    ----------
    model : Transformer
    src_text : 英语原文
    src_tok : 源语言分词器
    tgt_tok : 目标语言分词器
    device : 设备
    max_len : 最大解码长度

    Returns
    -------
    str : 德语翻译结果
    """
    # 编码源句子
    src_encoded = src_tok.encode(src_text)
    src_ids = torch.tensor([src_encoded.ids], dtype=torch.long, device=device)

    # 编码器前向
    src_padding_mask = torch.eq(src_ids, model.padding_idx)
    src_embedded = model.dropout1(model.positional_encoding1(model.encoder_embedding(src_ids)))
    enc_output = src_embedded
    for enc_layer in model.encoder_layers:
        enc_output = enc_layer(enc_output, src_padding_mask)

    # 自回归解码
    tgt_ids = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)

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
        next_token = logits.argmax(dim=-1).item()

        if next_token == EOS_ID:
            break

        tgt_ids = torch.cat(
            [tgt_ids, torch.tensor([[next_token]], dtype=torch.long, device=device)],
            dim=1,
        )

    # 解码输出 token IDs（跳过 BOS）
    output_ids = tgt_ids[0, 1:].tolist()
    output_ids = [t for t in output_ids if t not in (BOS_ID, EOS_ID, PAD_ID)]
    return tgt_tok.decode(output_ids)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI 入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="EN→DE 翻译器 (基于训练好的 Transformer)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="模型 checkpoint 路径 (默认: checkpoints/transformer_best.pt)",
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="要翻译的英文文本 (不指定则进入交互模式)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="运行设备 (默认: 自动选择 cuda/cpu)",
    )
    args = parser.parse_args()

    # 设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Checkpoint 路径
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = _CHECKPOINT_DIR / "transformer_best.pt"

    if not ckpt_path.exists():
        print(f"错误: 找不到模型文件 {ckpt_path}")
        print("请先运行 train.py 训练并保存模型:")
        print("  cd examples/transformer")
        print("  python train.py")
        sys.exit(1)

    # 加载
    print(f"加载模型: {ckpt_path}")
    model, cfg = load_model(ckpt_path, device)
    print(f"模型配置: {cfg['num_layers']}L-{cfg['d_model']}d-{cfg['num_heads']}h, "
          f"src_vocab={cfg['src_vocab_size']}, tgt_vocab={cfg['tgt_vocab_size']}")

    print("加载分词器 ...")
    src_tok, tgt_tok = load_tokenizers()
    print(f"设备: {device}")

    # 单句翻译模式
    if args.text:
        result = translate_sentence(model, args.text, src_tok, tgt_tok, device)
        print(f"\nEN: {args.text}")
        print(f"DE: {result}")
        return

    # 交互模式
    print("\n" + "═" * 60)
    print("  EN→DE 交互式翻译器")
    print("  输入英文句子，回车翻译。输入 quit 或 Ctrl+C 退出。")
    print("═" * 60 + "\n")

    while True:
        try:
            text = input("EN > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        result = translate_sentence(model, text, src_tok, tgt_tok, device)
        print(f"DE > {result}\n")


if __name__ == "__main__":
    main()
