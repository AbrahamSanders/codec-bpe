"""
Converter utility for converting discrete codec codes to and from unicode characters used for BPE tokenization.
"""
from typing import List, Optional, Union, Tuple
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)
UNICODE_OFFSET: int = 0x4E00

def codes_to_chars(
    codes: Union[List[List[int]], np.ndarray, torch.Tensor], 
    codebook_size: int,
    use_special_token_format: bool = False,
    copy_before_conversion: bool = True,
    unicode_offset: int = UNICODE_OFFSET,
) -> str:
    if isinstance(codes, list):
        codes = np.array(codes)
        copy_before_conversion = False
    elif isinstance(codes, torch.Tensor):
        codes = codes.cpu().numpy()
    if len(codes.shape) != 2:
        raise ValueError("codes must be a 2D array of shape (num_codebooks, seq_length).")
    if copy_before_conversion:
        codes = codes.copy()
    for i in range(codes.shape[0]):
        codes[i] += unicode_offset + i*codebook_size
    codes = codes.T.reshape(-1)
    chars = _flattened_shifted_codes_to_chars(codes, use_special_token_format)
    return chars

def chars_to_codes(
    chars: str, 
    num_codebooks: int,
    codebook_size: int,
    drop_inconsistent_codes: bool = True,
    drop_hanging_codes: bool = True,
    return_hanging_codes_chars: bool = False,
    return_tensors: Optional[str] = None, 
    unicode_offset: int = UNICODE_OFFSET,
) -> Union[List[List[int]], np.ndarray, torch.Tensor]:
    use_special_token_format = chars[0] == "<"
    if use_special_token_format:
        chars = chars.replace("<", "").replace(">", "")
    codes = np.array([ord(c) for c in chars])
    if drop_inconsistent_codes:
        codes = _drop_inconsistent_codes(codes, num_codebooks, codebook_size, unicode_offset)
    if drop_hanging_codes:
        codes, begin_hanging, end_hanging = _drop_hanging_codes(codes, num_codebooks, codebook_size, unicode_offset)
    codes = codes.reshape(-1, num_codebooks).T
    for i in range(codes.shape[0]):
        codes[i] -= unicode_offset + i*codebook_size
    if return_tensors is None:
        codes = codes.tolist()
    elif return_tensors == "pt":
        codes = torch.tensor(codes)
    if return_hanging_codes_chars:
        begin_hanging = _flattened_shifted_codes_to_chars(begin_hanging, use_special_token_format)
        end_hanging = _flattened_shifted_codes_to_chars(end_hanging, use_special_token_format)
        return codes, begin_hanging, end_hanging
    return codes

def _flattened_shifted_codes_to_chars(codes: np.ndarray, use_special_token_format: bool) -> str:
    if use_special_token_format:
        chars = "".join([f"<{chr(c)}>" for c in codes])
    else:
        chars = "".join([chr(c) for c in codes])
    return chars

def _resolve_codebook(code: int, num_codebooks: int, codebook_size: int, unicode_offset: int) -> int:
    codebook = num_codebooks-1
    while codebook > -1 and code < unicode_offset + codebook*codebook_size:
        codebook -= 1
    return codebook

def _drop_inconsistent_codes(
    codes: np.ndarray, 
    num_codebooks: int,
    codebook_size: int,
    unicode_offset: int,
) -> np.ndarray:
    mask = np.ones_like(codes, dtype=bool)
    expected_codebook = _resolve_codebook(codes[0], num_codebooks, codebook_size, unicode_offset)
    if expected_codebook < 0:
        expected_codebook = 0
    for i in range(len(codes)):
        # figure out which codebook the character belongs to
        actual_codebook = _resolve_codebook(codes[i], num_codebooks, codebook_size, unicode_offset)
        # mark it to be dropped if it doesn't match the expected codebook
        if actual_codebook != expected_codebook:
            mask[i] = False
            logger.warning(
                f"Dropped inconsistent audio code at position {i}. "
                f"Expected codebook {expected_codebook} but got codebook {actual_codebook}."
            )
        else:
            expected_codebook = (expected_codebook + 1) % num_codebooks
    codes = codes[mask]
    return codes

def _drop_hanging_codes(
    codes: np.ndarray, 
    num_codebooks: int,
    codebook_size: int,
    unicode_offset: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # first check for hanging codes at the beginning
    begin_hanging = []
    while len(codes) > 0:
        actual_codebook = _resolve_codebook(codes[0], num_codebooks, codebook_size, unicode_offset)
        if actual_codebook == 0:
            break
        begin_hanging.append(codes[0])
        codes = codes[1:]
        logger.info(f"Dropped hanging audio code (codebook {actual_codebook}) at beginning of sequence.")
    # then check for hanging codes at the end
    end_hanging = []
    while len(codes) > 0:
        actual_codebook = _resolve_codebook(codes[-1], num_codebooks, codebook_size, unicode_offset)
        if actual_codebook == num_codebooks-1:
            break
        end_hanging.append(codes[-1])
        codes = codes[:-1]
        logger.info(f"Dropped hanging audio code (codebook {actual_codebook}) at end of sequence.")
    begin_hanging = np.array(begin_hanging)
    end_hanging = np.array(end_hanging)[::-1]
    return codes, begin_hanging, end_hanging


    