import argparse
import json
import os

from .tools.audio_encoder import AudioEncoder, CodecTypes, SUPPORTED_EXTENSIONS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert audio files to numpy files containing audio codes using a Codec"
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default="audio",
        help="Directory containing the audio files",
    )
    parser.add_argument(
        "--codes_path",
        type=str,
        default="output/codes",
        help="Directory to save the numpy codes files",
    )
    parser.add_argument(
        "--chunk_size_secs", 
        type=float, 
        default=30.0, help="Chunk size in seconds",
    )
    parser.add_argument(
        "--context_secs", 
        type=float, 
        default=0.0, 
        help=(
            "Context size in seconds for encoding (default: 0.0, no context). "
            "If set, chunks will be left-padded with max(0, context_secs-chunk_size_secs) "
            "seconds of previous audio, while only chunk_size_secs worth of codes will be saved. "
            "This is useful for codecs that require context for better encoding quality at "
            "very small chunk sizes."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of audio chunks to process in a single batch",
    )
    parser.add_argument(
        "--codec_type", 
        type=str, 
        choices=list(CodecTypes),
        default=None,
        help="Type of codec to use for encoding. None to infer the type from --codec_model.",
    )
    parser.add_argument(
        "--codec_model",
        type=str,
        default="facebook/encodec_24khz",
        help="Codec model path on the HuggingFace Model Hub.",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=None,
        help=(
            "Bandwidth for encoding. Only applies if --codec_type is 'encodec' or 'funcodec'. "
            "Values may be provided in kbps (e.g. 1.5) or in bps (e.g. 1500)."
            "For FunCodec, valid ranges for this parameter are listed in the 'Bitrate' column at "
            "https://github.com/modelscope/FunCodec?tab=readme-ov-file#available-models. "
            "For EnCodec, valid values are 1.5, 3.0, 6.0, 12.0, and 24.0 (kpbs). "
            "None uses the max bandwidth with FunCodec and the min bandwidth with EnCodec."
        ),
    )
    parser.add_argument(
        "--n_quantizers",
        type=int,
        default=None,
        help=(
            "Number of quantizers (codebooks) to use for encoding. None to use all quantizers. "
            "Only applies if --codec_type is 'dac' or 'mimi'."
        ),
    )
    parser.add_argument(
        "--stereo",
        action="store_true",
        help="Encode stereo audio channels separately instead of converting to mono",
    )
    parser.add_argument(
        "--file_per_chunk",
        action="store_true",
        help=(
            "Save each audio chunk as a separate numpy file with the start timestamp (secs) in the filename "
            "instead of the default behavior of concatenating all chunks into a single numpy file corresponding "
            "to the original audio file."
        ),
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=SUPPORTED_EXTENSIONS,
        help="Audio file extensions to convert. Formats must be supported by a librosa backend.",
    )
    parser.add_argument(
        "--audio_filter", 
        nargs="+",
        help=(
            "Audio file filters. If provided, file paths must match one of the filters to be converted."
        )
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite existing numpy codes directories. If not set, audio corresponding to existing "
            "numpy codes directories will be skipped."
        ),
    )
    parser.add_argument(
        "--codec_info_only",
        action="store_true",
        help="Only write codec info and do not convert any audio files.",
    )
    args = parser.parse_args()

    codec_name_for_path = args.codec_model.split("/")[-1]
    codec_setting_for_path = f"{args.chunk_size_secs}s_{args.context_secs}s"
    args.codes_path = os.path.join(
        args.codes_path, codec_name_for_path, codec_setting_for_path, "stereo" if args.stereo else "mono"
    )

    audio_encoder = AudioEncoder(
        args.codec_model,
        codec_type=args.codec_type,
        chunk_size_secs=args.chunk_size_secs,
        context_secs=args.context_secs,
        batch_size=args.batch_size,
        bandwidth=args.bandwidth,
        n_quantizers=args.n_quantizers,
        stereo=args.stereo,
        file_per_chunk=args.file_per_chunk,
    )

    codec_info = audio_encoder.get_codec_info()

    # iterate and convert
    if args.codec_info_only:
        os.makedirs(args.codes_path, exist_ok=True)
    else:
        result = audio_encoder.encode_audio(
            args.audio_path,
            args.codes_path,
            extensions=args.extensions,
            audio_filter=args.audio_filter,
            overwrite=args.overwrite,
        )
        # Print summary
        print(f"Attempted to convert {result.num_audio_files} audio files:")
        print(f"{result.num_audio_files-len(result.errored_audio_files)} Succeeded.")
        print(f"{len(result.errored_audio_files)} Errored.")
        print(f"{result.num_numpy_files} numpy files created.")
        print(f"{result.num_skipped_dirs} directories skipped.")
        if result.errored_audio_files:
            print("\nErrored files:")
            for file in result.errored_audio_files:
                print(file)

    # write codec info to the base codes directory
    codec_info_path = os.path.join(args.codes_path, "codec_info.json")
    with open(codec_info_path, "w") as f:
        json.dump(codec_info, f, indent=4)
    print("\nCodec info written.")
    print("\nDone.")
