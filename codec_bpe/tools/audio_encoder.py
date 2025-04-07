import librosa
import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import math
from typing import Optional, List, Union, Tuple, Dict
from transformers.feature_extraction_utils import BatchFeature
from enum import Enum
from tqdm import tqdm

class CodecTypes(Enum):
    ENCODEC = "encodec"
    DAC = "dac"
    MIMI = "mimi"
    FUNCODEC = "funcodec"
    XCODEC2 = "xcodec2"

    @classmethod
    def try_get_codec_type(cls, codec_model):
        codec_model = codec_model.lower()
        if "audio_codec" in codec_model:
            return cls.FUNCODEC
        if "encodec" in codec_model:
            return cls.ENCODEC
        if "dac" in codec_model:
            return cls.DAC
        if "mimi" in codec_model:
            return cls.MIMI
        if "xcodec2" in codec_model:
            return cls.XCODEC2
        raise ValueError(f"Could not infer codec type from codec model: {codec_model}. Please specify --codec_type.")

    def __str__(self):
        return self.value
    def __eq__(self, value):
        return str(self) == value

SUPPORTED_EXTENSIONS = [".mp3", ".wav", ".flac", ".opus"]

class DefaultProcessor:
    def __call__(self, raw_audio: Union[np.ndarray, List[np.ndarray]], sampling_rate: int, return_tensors: str = "pt") -> BatchFeature:
        if not isinstance(raw_audio, list):
            raw_audio = [raw_audio]
        # Process audio to get padded input tensor
        max_audio_len = max([audio.shape[-1] for audio in raw_audio])
        batch_tensors = [F.pad(torch.from_numpy(audio), (0, max_audio_len-audio.shape[-1])) for audio in raw_audio]
        inputs = BatchFeature(
            data={"input_values": torch.stack(batch_tensors).unsqueeze(1).float()}, 
            tensor_type=return_tensors,
        )
        return inputs

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
        chunk_size_secs: int = 60,
        batch_size: int = 1,
        bandwidth: Optional[float] = None,
        n_quantizers: Optional[int] = None,
        stereo: bool = False,
    ):
        self.codec_model = codec_model
        self.codec_type = codec_type
        if self.codec_type is None:
            self.codec_type = CodecTypes.try_get_codec_type(self.codec_model)
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk_size_secs = chunk_size_secs
        self.batch_size = batch_size
        self.bandwidth = bandwidth
        # support bandwidth in kbps or bps
        if self.bandwidth is not None:
            if self.codec_type == CodecTypes.FUNCODEC and self.bandwidth <= 16.0:
                self.bandwidth *= 1000
            if self.codec_type == CodecTypes.ENCODEC and self.bandwidth > 24.0:
                self.bandwidth /= 1000
        self.n_quantizers = n_quantizers
        self.stereo = stereo

        # load the codec model
        if self.codec_type == CodecTypes.FUNCODEC:
            from funcodec.bin.codec_inference import Speech2Token
            from huggingface_hub import snapshot_download
            cache_path = snapshot_download(self.codec_model)
            config_file = os.path.join(cache_path, "config.yaml")
            model_pth = os.path.join(cache_path, "model.pth")
            self.model = Speech2Token(config_file, model_pth, device=str(self.device))
            self.model.eval()
            self.processor = DefaultProcessor()
            self.sr = self.model.model_args.sampling_rate
        elif self.codec_type == CodecTypes.XCODEC2:
            if self.batch_size > 1:
                raise ValueError("XCodec2 only supports batch size 1 for now.")
            from xcodec2.modeling_xcodec2 import XCodec2Model
            self.model = XCodec2Model.from_pretrained(self.codec_model).to(self.device)
            self.processor = DefaultProcessor()
            self.sr = self.model.feature_extractor.sampling_rate
        else:
            from transformers import AutoModel, AutoProcessor
            self.model = AutoModel.from_pretrained(self.codec_model).to(self.device)
            self.processor = AutoProcessor.from_pretrained(self.codec_model)
            self.sr = self.model.config.sampling_rate

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

    def _process_batch(self, batch: List[np.ndarray], batch_info: List[Tuple[str, str, int, int]]) -> List[str]:
        errored_files = []
        if not batch:
            return errored_files
        
        try:        
            audio_codes, samples_per_frame = self._encode_batch(batch)
                
            # Save the non-padded part of the encoded audio
            batch_dim = 1 if self.codec_type == CodecTypes.ENCODEC else 0
            for i, (file_path, numpy_root, channel, start_secs) in enumerate(batch_info):
                encoded_chunk = audio_codes.select(batch_dim, i).unsqueeze(batch_dim)
                non_padded_len = math.ceil(batch[i].shape[-1] / samples_per_frame)
                encoded_chunk = encoded_chunk[..., :non_padded_len]

                # Save encoded chunk to numpy file
                file_name_noext = os.path.basename(os.path.splitext(file_path)[0])
                numpy_filepath = os.path.join(numpy_root, f"{file_name_noext}_c{channel}_t{start_secs:06d}.npy")
                os.makedirs(os.path.dirname(numpy_filepath), exist_ok=True)
                np.save(numpy_filepath, encoded_chunk.cpu().numpy(), allow_pickle=False)

        except Exception as e:
            print(f"Error encoding batch: {e}")
            errored_files.extend(set([info[0] for info in batch_info]))
        
        return errored_files

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
                    end = start + self.chunk_size_secs * self.sr
                    audio_chunk = audio[..., start:end]
                    if len(audio_chunk.shape) == 1:
                        audio_chunk = np.expand_dims(audio_chunk, axis=0)
                    for channel in range(audio_chunk.shape[0]):
                        batch.append(audio_chunk[channel])
                        batch_info.append((file_path, numpy_root, channel, start // self.sr))
                        
                        # Process batch if it reaches the specified size
                        if len(batch) == self.batch_size:
                            errored_files = self._process_batch(batch, batch_info)
                            result.num_numpy_files += len(batch) if not errored_files else 0
                            result.errored_audio_files.extend(errored_files)
                            batch.clear()
                            batch_info.clear()

                    if end >= audio.shape[-1]:
                        break
                    start = end

        # Process any remaining chunks in the batch
        if batch:
            errored_files = self._process_batch(batch, batch_info)
            result.num_numpy_files += len(batch) if not errored_files else 0
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
