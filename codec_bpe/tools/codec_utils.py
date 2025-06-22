import torch
import torch.nn.functional as F
import numpy as np
import os
from enum import Enum
from typing import Tuple, Union, List
from transformers.feature_extraction_utils import BatchFeature, FeatureExtractionMixin

MAGICODEC_MODELS = {
    "magicodec-50hz-base": {
        "ckpt": {
            "repo_id": "Ereboas/MagiCodec_16k_50hz",
            "filename": "MagiCodec-50Hz-Base.ckpt",
        },
    },
}

WAVTOKENIZER_MODELS = {
    "wavtokenizer-small-600-24k-4096": {
        "config": {
            "repo_id": "novateur/WavTokenizer",
            "filename": "wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        },
        "ckpt": {
            "repo_id": "novateur/WavTokenizer",
            "filename": "WavTokenizer_small_600_24k_4096.ckpt"
        },
    },
    "wavtokenizer-small-320-24k-4096": {
        "config": {
            "repo_id": "novateur/WavTokenizer",
            "filename": "wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        },
        "ckpt": {
            "repo_id": "novateur/WavTokenizer",
            "filename": "WavTokenizer_small_320_24k_4096.ckpt"
        },
    },
    "wavtokenizer-medium-speech-320-24k-4096": {
        "config": {
            "repo_id": "novateur/WavTokenizer-medium-speech-75token",
            "filename": "wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        },
        "ckpt": {
            "repo_id": "novateur/WavTokenizer-medium-speech-75token",
            "filename": "wavtokenizer_medium_speech_320_24k_v2.ckpt"
        },
    },
    "wavtokenizer-medium-music-audio-320-24k-4096": {
        "config": {
            "repo_id": "novateur/WavTokenizer-medium-music-audio-75token",
            "filename": "wavtokenizer_mediumdata_music_audio_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        },
        "ckpt": {
            "repo_id": "novateur/WavTokenizer-medium-music-audio-75token",
            "filename": "wavtokenizer_medium_music_audio_320_24k_v2.ckpt"
        },
    },
    "wavtokenizer-large-600-24k-4096": {
        "config": {
            "repo_id": "novateur/WavTokenizer",
            "filename": "wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        },
        "ckpt": {
            "repo_id": "novateur/WavTokenizer-large-unify-40token",
            "filename": "wavtokenizer_large_unify_600_24k.ckpt"
        },
    },
    "wavtokenizer-large-320-24k-4096": {
        "config": {
            "repo_id": "novateur/WavTokenizer",
            "filename": "wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        },
        "ckpt": {
            "repo_id": "novateur/WavTokenizer-large-speech-75token",
            "filename": "wavtokenizer_large_speech_320_v2.ckpt"
        },
    }
}

SIMVQ_MODELS = ["simvq_4k", "simvq_8k", "simvq_65k", "simvq_262k"]

class CodecTypes(Enum):
    ENCODEC = "encodec"
    DAC = "dac"
    MIMI = "mimi"
    FUNCODEC = "funcodec"
    XCODEC2 = "xcodec2"
    WAVTOKENIZER = "wavtokenizer"
    SIMVQ = "simvq"
    MAGICODEC = "magicodec"

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
        if "wavtokenizer" in codec_model:
            return cls.WAVTOKENIZER
        if "simvq" in codec_model:
            return cls.SIMVQ
        if "magicodec" in codec_model:
            return cls.MAGICODEC
        raise ValueError(f"Could not infer codec type from codec model: {codec_model}. Please specify --codec_type.")

    def __str__(self):
        return self.value
    def __eq__(self, value):
        return str(self) == value

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

def load_funcodec_model(codec_model: str, device: Union[str, torch.device]) -> Tuple[torch.nn.Module, DefaultProcessor, int]:
    from funcodec.bin.codec_inference import Speech2Token
    from huggingface_hub import snapshot_download
    cache_path = snapshot_download(codec_model)
    config_file = os.path.join(cache_path, "config.yaml")
    model_pth = os.path.join(cache_path, "model.pth")
    model = Speech2Token(config_file, model_pth, device=str(device))
    model.eval()
    processor = DefaultProcessor()
    sr = model.model_args.sampling_rate
    return model, processor, sr

def load_xcodec2_model(codec_model: str, device: Union[str, torch.device]) -> Tuple[torch.nn.Module, DefaultProcessor, int]:
    from huggingface_hub import hf_hub_download
    from xcodec2.modeling_xcodec2 import XCodec2Model
    from xcodec2.configuration_bigcodec import BigCodecConfig
    from safetensors import safe_open
    ckpt_path = hf_hub_download(repo_id=codec_model, filename="model.safetensors")
    ckpt = {}
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            ckpt[k.replace(".beta", ".bias")] = f.get_tensor(k)
    codec_config = BigCodecConfig.from_pretrained(codec_model)
    model = XCodec2Model.from_pretrained(None, config=codec_config, state_dict=ckpt)
    model = model.eval().to(device)
    processor = DefaultProcessor()
    sr = model.feature_extractor.sampling_rate
    return model, processor, sr

def load_wavtokenizer_model(codec_model: str, device: Union[str, torch.device]) -> Tuple[torch.nn.Module, DefaultProcessor, int]:
    # add `WavTokenizer` directory to the import path.
    # TODO: get rid of this if a proper WavTokenizer package is ever released.
    if not os.path.exists("WavTokenizer"):
        raise ValueError(
            "WavTokenizer not found in your working directory. Please clone the WavTokenizer repository: "
            "`git clone https://github.com/jishengpeng/WavTokenizer.git`"
        )
    import sys
    sys.path.append("WavTokenizer")
    from huggingface_hub import hf_hub_download
    from decoder.pretrained import WavTokenizer
    if codec_model.lower() not in WAVTOKENIZER_MODELS:
        raise ValueError(f"Unsupported wavtokenizer model: {codec_model}. Supported models: {list(WAVTOKENIZER_MODELS)}")
    model_info = WAVTOKENIZER_MODELS[codec_model.lower()]
    config_file = hf_hub_download(**model_info["config"])
    model_ckpt = hf_hub_download(**model_info["ckpt"])
    model = WavTokenizer.from_pretrained0802(config_file, model_ckpt).to(device)
    processor = DefaultProcessor()
    sr = model.feature_extractor.encodec.sample_rate
    return model, processor, sr

def load_simvq_model(codec_model: str, device: Union[str, torch.device]) -> Tuple[torch.nn.Module, DefaultProcessor, int]:
    # add `SimVQ` directory to the import path.
    # TODO: get rid of this if a proper SimVQ package is ever released.
    if not os.path.exists("SimVQ"):
        raise ValueError(
            "SimVQ not found in your working directory. Please clone the SimVQ repository: "
            "`git clone https://github.com/youngsheen/SimVQ.git`"
        )
    import sys
    sys.path.append("SimVQ")
    import importlib
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    if codec_model.lower() not in SIMVQ_MODELS:
        raise ValueError(f"Unsupported SimVQ model: {codec_model}. Supported models: {SIMVQ_MODELS}")
    config_file = hf_hub_download(repo_id="youngsheen/SimVQ", filename=f"vq_audio_log/{codec_model.lower()}/1second/config.yaml")
    model_ckpt = hf_hub_download(repo_id="youngsheen/SimVQ", filename=f"vq_audio_log/{codec_model.lower()}/epoch=49-step=138600.ckpt")
    config = OmegaConf.load(config_file)
    module, cls = config.model.class_path.rsplit(".", 1)
    cls_init = getattr(importlib.import_module(module, package=None), cls)
    model = cls_init(**config.model.init_args)
    sd = torch.load(model_ckpt, map_location="cpu")["state_dict"]
    model.load_state_dict(sd, strict=False)
    model = model.eval().to(device)
    processor = DefaultProcessor()
    sr = config.model.init_args.sample_rate
    return model, processor, sr

def load_magicodec_model(codec_model: str, device: Union[str, torch.device]) -> Tuple[torch.nn.Module, DefaultProcessor, int]:
    # add `MagiCodec` directory to the import path.
    # TODO: get rid of this if a proper MagiCodec package is ever released.
    if not os.path.exists("MagiCodec"):
        raise ValueError(
            "MagiCodec not found in your working directory. Please clone the MagiCodec repository: "
            "`git clone https://github.com/Ereboas/MagiCodec.git`"
        )
    import sys
    sys.path.append("MagiCodec")
    from huggingface_hub import hf_hub_download
    from codec.generator import Generator
    if codec_model.lower() not in MAGICODEC_MODELS:
        raise ValueError(f"Unsupported magicodec model: {codec_model}. Supported models: {list(MAGICODEC_MODELS)}")
    model_info = MAGICODEC_MODELS[codec_model.lower()]
    model_ckpt = hf_hub_download(**model_info["ckpt"])
    model = Generator(token_hz=50)
    state_dict = torch.load(model_ckpt, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model = model.eval().to(device)
    processor = DefaultProcessor()
    sr = model.sample_rate
    return model, processor, sr

def load_transformers_codec_model(codec_model: str, device: Union[str, torch.device]) -> Tuple[torch.nn.Module, FeatureExtractionMixin, int]:
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(codec_model).to(device)
    processor = AutoProcessor.from_pretrained(codec_model)
    sr = model.config.sampling_rate
    return model, processor, sr

def load_codec_model(
    codec_type: CodecTypes, 
    codec_model: str,
    device: Union[str, torch.device],
) -> Tuple[torch.nn.Module, Union[DefaultProcessor, FeatureExtractionMixin], int]:
    if codec_type == CodecTypes.FUNCODEC:
        return load_funcodec_model(codec_model, device)
    elif codec_type == CodecTypes.XCODEC2:
        return load_xcodec2_model(codec_model, device)
    elif codec_type == CodecTypes.WAVTOKENIZER:
        return load_wavtokenizer_model(codec_model, device)
    elif codec_type == CodecTypes.SIMVQ:
        return load_simvq_model(codec_model, device)
    elif codec_type == CodecTypes.MAGICODEC:
        return load_magicodec_model(codec_model, device)
    else:
        return load_transformers_codec_model(codec_model, device)