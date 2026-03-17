"""
zh_en_translate.py

中文→英文交互式翻译脚本。默认加载 zh_en_train.py 训练得到的模型。

用法::

    cd examples/transformer
    python zh_en_translate.py
    python zh_en_translate.py --text "今天天气很好。"
    python zh_en_translate.py --checkpoint checkpoints/transformer_zh_en_strat7.pt
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import Transformer

PAD_ID = 0
BOS_ID = 2
EOS_ID = 3

SCRIPT_DIR = Path(__file__).parent
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
DATA_DIR = SCRIPT_DIR / "data" / "zh_en"


def _normalize_zh_text(text: str) -> str:
    return " ".join(list(text.strip().replace(" ", "")))


def _normalize_en_text(text: str) -> str:
    return " ".join(text.strip().split())


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]

    model = Transformer(
        config["src_vocab_size"],
        config["tgt_vocab_size"],
        config["d_model"],
        config["num_heads"],
        config["num_layers"],
        config["d_ff"],
        config["max_seq_length"],
        config["dropout"],
        padding_idx=config["padding_idx"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


def load_tokenizers():
    src_path = DATA_DIR / "tokenizer_zh.json"
    tgt_path = DATA_DIR / "tokenizer_en.json"

    if not src_path.exists() or not tgt_path.exists():
        print("错误: 找不到中文-英文分词器文件。请先运行 zh_en_train.py。")
        print(f"  期望路径: {src_path}")
        print(f"  期望路径: {tgt_path}")
        sys.exit(1)

    src_tokenizer = Tokenizer.from_file(str(src_path))
    tgt_tokenizer = Tokenizer.from_file(str(tgt_path))
    return src_tokenizer, tgt_tokenizer


@torch.no_grad()
def translate_sentence(model, src_text: str, src_tok, tgt_tok, device, max_len: int = 64):
    normalized_text = _normalize_zh_text(src_text)
    src_ids = src_tok.encode(normalized_text).ids[:max_len]
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    src_padding_mask = torch.eq(src_tensor, model.padding_idx)
    src_embedded = model.dropout1(model.positional_encoding1(model.encoder_embedding(src_tensor)))
    enc_output = src_embedded
    for encoder_layer in model.encoder_layers:
        enc_output = encoder_layer(enc_output, src_padding_mask)

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
        for decoder_layer in model.decoder_layers:
            dec_output = decoder_layer(
                dec_output,
                enc_output,
                causal_mask,
                src_padding_mask,
                tgt_padding_mask,
            )

        logits = model.fc(dec_output[:, -1, :])
        next_token = logits.argmax(dim=-1).item()
        if next_token == EOS_ID:
            break

        tgt_ids = torch.cat(
            [tgt_ids, torch.tensor([[next_token]], dtype=torch.long, device=device)],
            dim=1,
        )

    output_ids = tgt_ids[0, 1:].tolist()
    output_ids = [token for token in output_ids if token not in (BOS_ID, EOS_ID, PAD_ID)]
    return _normalize_en_text(tgt_tok.decode(output_ids))


def main():
    parser = argparse.ArgumentParser(description="ZH→EN 翻译器")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="模型 checkpoint 路径，默认读取 checkpoints/transformer_zh_en_best.pt",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="要翻译的中文句子，不指定则进入交互模式",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="运行设备，默认自动选择 cuda/cpu",
    )
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = CHECKPOINT_DIR / "transformer_zh_en_best.pt"

    if not checkpoint_path.exists():
        print(f"错误: 找不到模型文件 {checkpoint_path}")
        print("请先运行 zh_en_train.py 训练并保存模型。")
        sys.exit(1)

    print(f"加载模型: {checkpoint_path}")
    model, config = load_model(checkpoint_path, device)
    print(
        f"模型配置: {config['num_layers']}L-{config['d_model']}d-{config['num_heads']}h, "
        f"src_vocab={config['src_vocab_size']}, tgt_vocab={config['tgt_vocab_size']}"
    )

    print("加载分词器 ...")
    src_tok, tgt_tok = load_tokenizers()
    print(f"设备: {device}")

    if args.text:
        result = translate_sentence(model, args.text, src_tok, tgt_tok, device, config["max_seq_length"])
        print(f"\nZH: {args.text}")
        print(f"EN: {result}")
        return

    print("\n" + "═" * 60)
    print("  ZH→EN 交互式翻译器")
    print("  输入中文句子，回车翻译。输入 quit 或 Ctrl+C 退出。")
    print("═" * 60 + "\n")

    while True:
        try:
            text = input("ZH > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        result = translate_sentence(model, text, src_tok, tgt_tok, device, config["max_seq_length"])
        print(f"EN > {result}\n")


if __name__ == "__main__":
    main()