from tqdm import tqdm
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute statistics for a plain-text codec BPE dataset")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--num_codebooks", type=int, required=True)
    parser.add_argument("--codec_framerate", type=float, required=True)
    parser.add_argument("--audio_start_token", type=str)
    parser.add_argument("--audio_end_token", type=str)
    parser.add_argument("--num_examples", type=int, default=None)
    args = parser.parse_args()

    lengths = []
    with open(args.dataset_path, encoding="utf-8") as f:
        for i, line in tqdm(enumerate(f), desc="Examples"):
            if i == args.num_examples:
                break
            line = line.rstrip()
            if args.audio_start_token is not None:
                line = line.lstrip(args.audio_start_token)
            if args.audio_end_token is not None:
                line = line.rstrip(args.audio_end_token)
            if line[0] == "<":
                line = line.replace("<", "").replace(">", "")
            num_units = len(line) / args.num_codebooks
            num_seconds = num_units / args.codec_framerate
            lengths.append(num_seconds)
    total_seconds = np.sum(lengths)

    print(f"{len(lengths)} examples")
    print(f"Total: {total_seconds:.2f} seconds ({(total_seconds / 3600):.2f} hours)")
    print(f"Max: {np.max(lengths):.2f} seconds")
    print(f"Min: {np.min(lengths):.2f} seconds")
    print(f"Median: {np.median(lengths):.2f} seconds")
    print(f"Mean: {np.mean(lengths):.2f} seconds")
    print(f"Std: {np.std(lengths):.2f} seconds")
