import argparse
from transformers import AutoTokenizer

from .tools.extender import extend_existing_tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extend an existing Transformers tokenizer with codec BPE tokens")
    parser.add_argument("--existing_tokenizer", type=str, required=True)
    parser.add_argument("--codec_bpe_tokenizer", type=str, required=True)
    parser.add_argument("--audio_start_token", type=str)
    parser.add_argument("--audio_end_token", type=str)
    parser.add_argument("--use_special_token_format", action="store_true")
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = f"output/{args.existing_tokenizer}_extended"

    existing_tokenizer = AutoTokenizer.from_pretrained(args.existing_tokenizer)
    codec_bpe_tokenizer = AutoTokenizer.from_pretrained(args.codec_bpe_tokenizer)

    num_added = extend_existing_tokenizer(
        existing_tokenizer,
        codec_bpe_tokenizer,
        args.audio_start_token,
        args.audio_end_token,
        args.use_special_token_format,
    )
    print(f"Added {num_added} tokens to the existing tokenizer {args.existing_tokenizer} and saved it as {args.save_path}.")
    existing_tokenizer.save_pretrained(args.save_path)