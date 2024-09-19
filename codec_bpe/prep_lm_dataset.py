import argparse
import os
from tqdm import tqdm
from transformers import AutoTokenizer

from .tools.lm_dataset_builder import LMDatasetBuilder
from . import UNICODE_OFFSET

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use numpy files containing audio codes to construct a plain-text codec BPE dataset suitable for language modeling"
    )
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--codes_path", type=str, required=True)
    parser.add_argument("--num_codebooks", type=int, required=True)
    parser.add_argument("--codebook_size", type=int, required=True)
    parser.add_argument("--audio_start_token", type=str)
    parser.add_argument("--audio_end_token", type=str)
    parser.add_argument("--use_special_token_format", action="store_true")
    parser.add_argument("--unicode_offset", type=int, default=UNICODE_OFFSET)
    parser.add_argument("--sequence_length", type=int, default=4096)
    parser.add_argument("--overlap_length", type=int, default=1024)
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument("--save_path", type=str, default="output/lm_dataset.txt")
    parser.add_argument("--codes_filter", type=str, nargs="+")
    parser.add_argument("--num_examples", type=int, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    lm_dataset_builder = LMDatasetBuilder(
        tokenizer=tokenizer,
        num_codebooks=args.num_codebooks,
        codebook_size=args.codebook_size,
        audio_start_token=args.audio_start_token,
        audio_end_token=args.audio_end_token,
        use_special_token_format=args.use_special_token_format,
        unicode_offset=args.unicode_offset,
        sequence_length=args.sequence_length,
        overlap_length=args.overlap_length,
        drop_last=args.drop_last,
    )

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with open(args.save_path, "w", encoding="utf-8") as f:
        for i, example in tqdm(enumerate(lm_dataset_builder.iterate_examples(args.codes_path, args.codes_filter)), desc="Examples"):
            if i == args.num_examples:
                break
            f.write(example)
            f.write("\n")
            