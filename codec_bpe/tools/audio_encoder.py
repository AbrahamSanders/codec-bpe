import librosa
import os
import shutil
import numpy as np
import torch
import math
from typing import Optional, List, Union, Tuple, Dict
from tqdm import tqdm

from .codec_utils import CodecTypes, load_codec_model

SUPPORTED_EXTENSIONS = [".mp3", ".wav", ".flac", ".opus"]

class AudioEncodeResult:
    def __init__(self):
        self.num_audio_files = 0
        self.num_numpy_files = 0
        self.num_skipped_dirs = 0
        self.errored_audio_files = []

class AudioEncoder:
    def __init__(
        self, 
        codec_model: str,
        codec_type: Optional[CodecTypes] = None,
        device: Optional[Union[str, torch.device]] = None,
        chunk_size_secs: float = 30.0,
        context_secs: float = 0.0,
        batch_size: int = 1,
        bandwidth: Optional[float] = None,
        n_quantizers: Optional[int] = None,
        stereo: bool = False,
        file_per_chunk: bool = False,
    ):
        self.codec_model = codec_model
        self.codec_type = codec_type
        if self.codec_type is None:
            self.codec_type = CodecTypes.try_get_codec_type(self.codec_model)
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(self.device, str):
            self.device = torch.device(self.device)
        self.chunk_size_secs = chunk_size_secs
        self.context_secs = context_secs
        self.batch_size = batch_size
        if self.batch_size > 1 and self.codec_type == CodecTypes.XCODEC2:
            raise ValueError("XCodec2 only supports batch size 1 for now.")
        self.bandwidth = bandwidth
        # support bandwidth in kbps or bps
        if self.bandwidth is not None:
            if self.codec_type == CodecTypes.FUNCODEC and self.bandwidth <= 16.0:
                self.bandwidth *= 1000
            if self.codec_type == CodecTypes.ENCODEC and self.bandwidth > 24.0:
                self.bandwidth /= 1000
        self.n_quantizers = n_quantizers
        self.stereo = stereo
        self.file_per_chunk = file_per_chunk

        # load the codec model
        self.model, self.processor, self.sr = load_codec_model(self.codec_type, self.codec_model, self.device)
        self.chunk_size_samples = int(self.chunk_size_secs * self.sr)
        self.context_samples = int(max(0.0, self.context_secs-self.chunk_size_secs) * self.sr)

    def _encode_batch(self, batch: List[np.ndarray]) -> Tuple[torch.Tensor, float]:
        # Process audio to get padded input tensor
        inputs = self.processor(raw_audio=batch, sampling_rate=self.sr, return_tensors="pt").to(self.device)
        input_values = inputs.input_values

        # Encode the batch
        with torch.no_grad():
            if self.codec_type == CodecTypes.FUNCODEC:
                encoded_batch, _, _, _ = self.model(
                    input_values,
                    bit_width=int(self.bandwidth) if self.bandwidth is not None else None,
                    run_mod="encode",
                )
                # Permute dimensions to match expected format
                audio_codes = torch.permute(encoded_batch[0], (1, 0, 2))
            elif self.codec_type == CodecTypes.XCODEC2:
                input_values = input_values.squeeze(1)
                audio_codes = self.model.encode_code(input_values, sample_rate=self.sr)
            elif self.codec_type == CodecTypes.WAVTOKENIZER:
                input_values = input_values.squeeze(1)
                bandwidth_id = torch.tensor([0]).to(self.device)
                _, audio_codes = self.model.encode_infer(input_values, bandwidth_id=bandwidth_id)
                # Permute dimensions to match expected format
                audio_codes = torch.permute(audio_codes, (1, 0, 2))
            elif self.codec_type == CodecTypes.SIMVQ:
                _, _, audio_codes, _ = self.model.encode(input_values)
                audio_codes = audio_codes.view(input_values.shape[0], 1, -1)
            elif self.codec_type == CodecTypes.MAGICODEC:
                with torch.autocast(
                    device_type = "cuda",
                    dtype = torch.bfloat16,
                    enabled = self.device.type == "cuda" and torch.cuda.is_bf16_supported(),
                ):
                    x = self.model.pad_audio(input_values)
                    z_e = self.model.encoder(x)
                    _, audio_codes = self.model.quantizer.inference(z_e)
                audio_codes = audio_codes.unsqueeze(1)
            else:
                encode_kwargs = {}
                if self.codec_type == CodecTypes.DAC:
                    encode_kwargs["n_quantizers"] = self.n_quantizers
                elif self.codec_type == CodecTypes.MIMI:
                    encode_kwargs["num_quantizers"] = self.n_quantizers
                elif self.codec_type == CodecTypes.ENCODEC:
                    encode_kwargs["bandwidth"] = self.bandwidth
                outputs = self.model.encode(**inputs, **encode_kwargs)
                audio_codes = outputs.audio_codes
        
        samples_per_frame = math.ceil(input_values.shape[-1] / audio_codes.shape[-1])
        return audio_codes, samples_per_frame

    def _process_batch(
        self, 
        batch: List[np.ndarray], 
        batch_info: List[Tuple[str, str, int, float, bool]], 
        encoded_file_chunks: List[List[np.ndarray]],
    ) -> List[str]:
        errored_files = []
        num_numpy_files = 0
        if not batch:
            return errored_files, num_numpy_files
        
        try:        
            audio_codes, samples_per_frame = self._encode_batch(batch)
                
            # Save the non-padded part of the encoded audio
            batch_dim = 1 if self.codec_type == CodecTypes.ENCODEC else 0
            for i, (file_path, numpy_root, channel, start_secs, end_of_file) in enumerate(batch_info):
                encoded_chunk = audio_codes.select(batch_dim, i).unsqueeze(batch_dim)
                context_len = math.ceil(self.context_samples / samples_per_frame)
                non_padded_len = math.ceil(batch[i].shape[-1] / samples_per_frame)
                encoded_chunk = encoded_chunk[..., context_len:non_padded_len]

                # Save encoded chunk to numpy file
                if not self.file_per_chunk:
                    encoded_file_chunks[channel].append(encoded_chunk.cpu().numpy())
                if self.file_per_chunk or end_of_file:
                    file_name_noext = os.path.basename(os.path.splitext(file_path)[0])
                    start_secs_whole = int(start_secs)
                    start_secs_ms = round((start_secs - start_secs_whole) * 1000)
                    timestamp_slot = f"_t{start_secs_whole:06d}_{start_secs_ms:03d}" if self.file_per_chunk else ""
                    numpy_filepath = os.path.join(numpy_root, f"{file_name_noext}_c{channel}{timestamp_slot}.npy")
                    os.makedirs(os.path.dirname(numpy_filepath), exist_ok=True)
                    if self.file_per_chunk:
                        np.save(numpy_filepath, encoded_chunk.cpu().numpy(), allow_pickle=False)
                    else:
                        encoded_file = np.concatenate(encoded_file_chunks[channel], axis=-1)
                        np.save(numpy_filepath, encoded_file, allow_pickle=False)
                        encoded_file_chunks[channel].clear()
                    num_numpy_files += 1

        except Exception as e:
            print(f"Error encoding batch: {e}")
            errored_files.extend(set([info[0] for info in batch_info]))
        
        return num_numpy_files, errored_files

    def encode_audio(
        self,
        audio_path: str,
        codes_path: str,
        extensions: List[str] = SUPPORTED_EXTENSIONS,
        audio_filter: Optional[Union[str, List[str]]] = None,
        overwrite: bool = False,
    ) -> AudioEncodeResult:
        # traverse the audio directory recursively and convert in each subdirectory containing
        # audio fileswith the specified extensions
        if isinstance(audio_filter, str):
            audio_filter = [audio_filter]
        result = AudioEncodeResult()
        batch = []
        batch_info = []
        encoded_file_chunks = [[], []] if self.stereo else [[]]
        for root, _, files in os.walk(audio_path):
            files = sorted([os.path.join(root, f) for f in files if os.path.splitext(f)[1] in extensions])
            if audio_filter:
                files = [f for f in files if any([filter_ in f for filter_ in audio_filter])]
            if len(files) == 0:
                continue
            numpy_root = root.replace(audio_path, codes_path)
            if os.path.exists(numpy_root):
                if overwrite:
                    shutil.rmtree(numpy_root)
                else:
                    print(f"Skipping {root} because {numpy_root} already exists.")
                    result.num_skipped_dirs += 1
                    continue
            print(f"Converting in {root}...")
            for file_path in tqdm(files, desc="Files"):
                result.num_audio_files += 1
                try:
                    # Load the audio file
                    audio, _ = librosa.load(file_path, sr=self.sr, mono=not self.stereo)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    result.errored_audio_files.append(file_path)
                    continue

                # Encode it in chunks of size chunk_size_secs on each channel independently
                start = 0
                while True:
                    end = start + self.chunk_size_samples
                    end_of_file = end >= audio.shape[-1]
                    start_with_context = start - self.context_samples
                    audio_chunk = audio[..., max(0, start_with_context):end]
                    if audio_chunk.ndim == 1:
                        audio_chunk = np.expand_dims(audio_chunk, axis=0)
                    if start_with_context < 0:
                        # if we are at the beginning of the audio, pad with a silent context
                        audio_chunk = np.pad(audio_chunk, ((0, 0), (-start_with_context, 0)), mode='constant')
                    for channel in range(audio_chunk.shape[0]):
                        batch.append(audio_chunk[channel])
                        batch_info.append((file_path, numpy_root, channel, start / self.sr, end_of_file))
                        
                        # Process batch if it reaches the specified size
                        if len(batch) == self.batch_size:
                            num_numpy_files, errored_files = self._process_batch(batch, batch_info, encoded_file_chunks)
                            result.num_numpy_files += num_numpy_files
                            result.errored_audio_files.extend(errored_files)
                            batch.clear()
                            batch_info.clear()

                    if end_of_file:
                        break
                    start = end

        # Process any remaining chunks in the batch
        if batch:
            num_numpy_files, errored_files = self._process_batch(batch, batch_info, encoded_file_chunks)
            result.num_numpy_files += num_numpy_files
            result.errored_audio_files.extend(errored_files)

        result.errored_audio_files = sorted(set(result.errored_audio_files))
        return result

    def get_codec_info(self) -> Dict[str, Union[str, int, float]]:
        # encode ten seconds of audio and get the number of codebooks and framerate
        dummy_audio = np.zeros(10 * self.sr)
        audio_codes, samples_per_frame = self._encode_batch([dummy_audio])
        # get stats
        if self.codec_type == CodecTypes.FUNCODEC:
            codebook_size = self.model.model_args.quantizer_conf["codebook_size"]
        elif self.codec_type == CodecTypes.XCODEC2:
            codebook_size = 65536
        elif self.codec_type == CodecTypes.WAVTOKENIZER:
            codebook_size = self.model.feature_extractor.encodec.quantizer.bins
        elif self.codec_type == CodecTypes.SIMVQ:
            codebook_size = self.model.quantize.n_e
        elif self.codec_type == CodecTypes.MAGICODEC:
            codebook_size = self.model.codebook_size
        else:
            codebook_size = self.model.config.codebook_size

        # write codec info to json
        codec_info = {
            "codec_type": str(self.codec_type),
            "codec_model": self.codec_model,
            "sampling_rate": self.sr,
            "num_codebooks": audio_codes.shape[-2],
            "codebook_size": codebook_size,
            "framerate": self.sr / samples_per_frame,
        }
        return codec_info
