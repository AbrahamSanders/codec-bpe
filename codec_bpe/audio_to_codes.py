import librosa
import os
import shutil
import argparse
import numpy as np
import torch
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert audio files to numpy files containing Encodec or DAC codes")
    parser.add_argument("--audio_path", type=str, default="audio", help="Directory containing the audio files")
    parser.add_argument("--codes_path", type=str, default="output/codes", help="Directory to save the numpy codes files")
    parser.add_argument("--chunk_size_secs", type=int, default=60, help="Chunk size in seconds")
    parser.add_argument("--encodec_model", type=str, default="facebook/encodec_24khz", help="Encodec model name. Ignored if --use_dac or --use_mimi are set.")
    parser.add_argument("--bandwidth", type=float, default=6.0, help="Bandwidth for encoding. Ignored if --use_dac or --use_mimi are set.")
    parser.add_argument("--dac_model", type=str, default="16khz", help="DAC model name. Only applies if --use_dac is set.")
    parser.add_argument("--mimi_model", type=str, default="kyutai/mimi", help="Mimi model name. Only applies if --use_mimi is set.")
    parser.add_argument("--n_quantizers", type=int, default=None, help="Number of quantizers for DAC model. None to use all quantizers. Only applies if --use_dac or --use_mimi are set.")
    parser.add_argument("--use_dac", action="store_true", help="Use DAC model instead of Encodec model")
    parser.add_argument("--use_mimi", action="store_true", help="Use Mimi model instead of Encodec model")
    parser.add_argument("--stereo", action="store_true", help="Encode stereo audio channels separately instead of converting to mono")
    parser.add_argument("--extensions", nargs="+", default=[".mp3", ".wav", ".flac", ".opus"], help="Audio file extensions to convert. Formats must be supported by a librosa backend.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing numpy codes directories. If not set, audio corresponding to existing numpy codes directories will be skipped.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.use_dac:
        import dac
        from audiotools import AudioSignal
        model_path = dac.utils.download(model_type=args.dac_model)
        model = dac.DAC.load(model_path).to(device)
        model.eval()
        codec_name_for_path = f"dac_{args.dac_model}"
    elif args.use_mimi:
        from transformers import MimiModel, AutoProcessor
        model = MimiModel.from_pretrained(args.mimi_model).to(device)
        processor = AutoProcessor.from_pretrained(args.mimi_model)
        codec_name_for_path = args.mimi_model.split("/")[-1]
    else:
        from transformers import EncodecModel, AutoProcessor
        model = EncodecModel.from_pretrained(args.encodec_model).to(device)
        processor = AutoProcessor.from_pretrained(args.encodec_model)
        codec_name_for_path = args.encodec_model.split("/")[-1]

    codes_path = os.path.join(args.codes_path, codec_name_for_path, "stereo" if args.stereo else "mono")
    num_audio_files = 0
    num_converted_audio_files = 0
    num_numpy_files = 0
    num_skipped_dirs = 0
    for root, dirs, files in os.walk(args.audio_path):
        files = sorted([f for f in files if os.path.splitext(f)[1] in args.extensions])
        if len(files) == 0:
            continue
        numpy_root = root.replace(args.audio_path, codes_path)
        if os.path.exists(numpy_root):
            if args.overwrite:
                shutil.rmtree(numpy_root)
            else:
                print(f"Skipping {root} because {numpy_root} already exists.")
                num_skipped_dirs += 1
                continue
        print(f"Converting in {root}...")
        for file in tqdm(files, desc="Files"):
            file_path = os.path.join(root, file)
            file_name_noext, _ = os.path.splitext(file)
            try:
                # Load the audio file
                num_audio_files += 1
                sr = model.sample_rate if args.use_dac else model.config.sampling_rate
                audio, sr = librosa.load(file_path, sr=sr, mono=not args.stereo)

                start = 0
                while True:
                    end = start + args.chunk_size_secs * sr
                    audio_chunk = audio[..., start:end]
                    if len(audio_chunk.shape) == 1:
                        audio_chunk = np.expand_dims(audio_chunk, axis=0)
                    for channel in range(audio_chunk.shape[0]):
                        channel_chunk = audio_chunk[channel]
                        if args.use_dac:
                            # prepare for model
                            signal = AudioSignal(channel_chunk, sample_rate=sr).to(device)
                            inputs = model.preprocess(signal.audio_data, signal.sample_rate)
                            # encode
                            with torch.no_grad():
                                _, encoded_chunk, _, _, _ = model.encode(inputs, n_quantizers=args.n_quantizers)
                        elif args.use_mimi:
                            # prepare for model
                            inputs = processor(raw_audio=channel_chunk, sampling_rate=sr, return_tensors="pt").to(device)
                            # encode
                            with torch.no_grad():
                                encoded_chunk = model.encode(**inputs, num_quantizers=args.n_quantizers).audio_codes
                        else:
                            # prepare for model
                            inputs = processor(raw_audio=channel_chunk, sampling_rate=sr, return_tensors="pt").to(device)
                            # encode
                            with torch.no_grad():
                                encoded_chunk = model.encode(**inputs, bandwidth=args.bandwidth).audio_codes

                        # Save the numpy file
                        start_secs = start // sr
                        numpy_filepath = os.path.join(numpy_root, f"{file_name_noext}_c{channel}_t{start_secs:06d}.npy")
                        os.makedirs(os.path.dirname(numpy_filepath), exist_ok=True)
                        np.save(numpy_filepath, encoded_chunk.cpu().numpy(), allow_pickle=False)
                        num_numpy_files += 1

                    if end >= audio.shape[-1]:
                        break
                    start = end
                num_converted_audio_files += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print(f"Attempted to convert {num_audio_files} audio files:")
    print(f"{num_converted_audio_files} Succeeded.")
    print(f"{num_audio_files-num_converted_audio_files} Failed.")
    print(f"{num_numpy_files} numpy files created.")
    print(f"{num_skipped_dirs} directories skipped.")
