import argparse
import os
from codec_bpe import Trainer, UNICODE_OFFSET

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a codec BPE tokenizer from numpy files containing audio codes")
    parser.add_argument("--codes_path", type=str, required=True)
    parser.add_argument("--num_codebooks", type=int, required=True)
    parser.add_argument("--codebook_size", type=int, required=True)
    parser.add_argument("--codec_framerate", type=int, default=None)
    parser.add_argument("--chunk_size_secs", type=int, default=None)
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--special_tokens", nargs="+", default=None)
    parser.add_argument("--unk_token", type=str, default=None)
    parser.add_argument("--max_token_codebook_ngrams", type=int, default=None)
    parser.add_argument("--unicode_offset", type=int, default=UNICODE_OFFSET)
    parser.add_argument("--save_path", type=str, default="output/tokenizer.json")
    parser.add_argument("--num_files", type=int, default=None)
    args = parser.parse_args()

    trainer = Trainer(
        args.num_codebooks,
        args.codebook_size,
        args.codec_framerate,
        args.chunk_size_secs,
        args.vocab_size,
        args.min_frequency,
        args.special_tokens,
        args.unk_token,
        args.max_token_codebook_ngrams,
        args.unicode_offset,
    )
    tokenizer = trainer.train(args.codes_path, args.num_files)
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(args.save_path)
