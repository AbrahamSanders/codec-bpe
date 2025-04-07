import argparse
import functools

from .core.trainer import Trainer
from .core.utils import get_codec_info, update_args_from_codec_info
from . import UNICODE_OFFSET

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a codec BPE tokenizer from numpy files containing audio codes")
    parser.add_argument("--codes_path", type=str, required=True)
    parser.add_argument("--num_codebooks", type=int, default=None)
    parser.add_argument("--codebook_size", type=int, default=None)
    parser.add_argument("--codec_framerate", type=float, default=None)
    parser.add_argument("--chunk_size_secs", type=int, default=None)
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--special_tokens", nargs="+", default=None)
    parser.add_argument("--bos_token", type=str)
    parser.add_argument("--eos_token", type=str)
    parser.add_argument("--unk_token", type=str)
    parser.add_argument("--pad_token", type=str)
    parser.add_argument("--max_token_codebook_ngrams", type=int, default=None)
    # handle hex values for unicode_offset with argparse: https://stackoverflow.com/a/25513044
    parser.add_argument("--unicode_offset", type=functools.partial(int, base=0), default=UNICODE_OFFSET)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--codes_filter", type=str, nargs="+")
    parser.add_argument("--num_files", type=int, default=None)
    args = parser.parse_args()

    codec_info = get_codec_info(args.codes_path)
    update_args_from_codec_info(args, codec_info)
    if args.num_codebooks is None or args.codebook_size is None:
        raise ValueError(
            "codec_info.json does not exist in --codes_path so you must specify --num_codebooks and --codebook_size manually."
        )

    codec_type = codec_info["codec_type"] if codec_info is not None else "codec"
    if args.save_path is None:
        args.save_path = f"output/{codec_type}_bpe_{args.num_codebooks}cb_{round(args.vocab_size/1000)}k"

    trainer = Trainer(
        args.num_codebooks,
        args.codebook_size,
        args.codec_framerate,
        args.chunk_size_secs,
        args.vocab_size,
        args.min_frequency,
        args.special_tokens,
        args.bos_token,
        args.eos_token,
        args.unk_token,
        args.pad_token,
        args.max_token_codebook_ngrams,
        args.unicode_offset,
    )
    tokenizer = trainer.train(args.codes_path, args.codes_filter, args.num_files)
    tokenizer.save_pretrained(args.save_path)
