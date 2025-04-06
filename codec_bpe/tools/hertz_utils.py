# hertz-dev is not an installable package, so we need to download it and add it to the path
import os
import torch
import sys
if not os.path.exists("hertz-dev"):
    from .zip_utils import download_and_extract_zip
    download_and_extract_zip("https://github.com/Standard-Intelligence/hertz-dev/archive/refs/heads/main.zip")
    os.rename("hertz-dev-main", "hertz-dev")
if "hertz-dev" not in sys.path:
    sys.path.append("hertz-dev")

from model import get_hertz_dev_config
from tokenizer import make_tokenizer

class HertzConfig:
    def __init__(self):
        self.hertz_dev_config = get_hertz_dev_config()
        self.sampling_rate = 16000
        self.codebook_size = self.hertz_dev_config.vocab_size

class HertzCodec:
    def __init__(self, device):
        self.config = HertzConfig()
        self.tokenizer = make_tokenizer("cpu")
        self.quantizer = self.make_quantizer()
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.tokenizer = self.tokenizer.to(self.dtype).to(device)
        self.quantizer = self.quantizer.to(self.dtype).to(device)

    def make_quantizer(self):
        quantizer = self.config.hertz_dev_config.quantizer_config().eval()
        quantizer.requires_grad = False
        return quantizer

    @torch.no_grad()
    def encode(self, audio):
        audio = audio.to(torch.float32).to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            encoder_outputs = self.tokenizer.latent_from_data(audio) # (batch_size, seq_len, latent_dim)
            _, audio_codes = self.quantizer(encoder_outputs, return_latent=True) # (batch_size, seq_len)
            audio_codes = audio_codes.unsqueeze(1) # (batch_size, num_codebooks, seq_len)
            return audio_codes
