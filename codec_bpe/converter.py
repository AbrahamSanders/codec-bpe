"""
Converter utility for converting discrete codec codes to and from unicode characters used for BPE tokenization.
"""
from typing import List, Optional, Union
import numpy as np
import torch

UNICODE_OFFSET: int = 0x4E00

def codes_to_chars(
    codes: Union[List[List[int]], np.ndarray, torch.Tensor], 
    codebook_size: int,
    unicode_offset: int = UNICODE_OFFSET,
) -> str:
    if isinstance(codes, list):
        codes = np.array(codes)
    elif isinstance(codes, torch.Tensor):
        codes = codes.cpu().numpy()
    if len(codes.shape) != 2:
        raise ValueError("codes must be a 2D array of shape (num_codebooks, seq_length).")
    for i in range(codes.shape[0]):
        codes[i] += unicode_offset + i*codebook_size
    codes = codes.T.reshape(-1)
    chars = "".join([chr(c) for c in codes])
    return chars

def chars_to_codes(
    chars: str, 
    num_codebooks: int,
    codebook_size: int,
    return_tensors: Optional[str] = None, 
    unicode_offset: int = UNICODE_OFFSET,
) -> Union[List[List[int]], np.ndarray, torch.Tensor]:
    codes = np.array([ord(c) for c in chars])
    codes = codes.reshape(-1, num_codebooks).T
    for i in range(codes.shape[0]):
        codes[i] -= unicode_offset + i*codebook_size
    if return_tensors is None:
        codes = codes.tolist()
    elif return_tensors == "pt":
        codes = torch.tensor(codes)
    return codes
    