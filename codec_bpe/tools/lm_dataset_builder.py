from typing import Optional, Union, Iterator, List
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from tqdm import tqdm
import numpy as np
import re

from ..core.converter import codes_to_chars, validate_unicode_offset, UNICODE_OFFSET
from ..core.utils import get_codes_files

class LMDatasetBuilder:
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        num_codebooks: int,
        codebook_size: int,
        audio_start_token: Optional[str] = None,
        audio_end_token: Optional[str] = None,
        unicode_offset: int = UNICODE_OFFSET,
        sequence_length: int = 4096,
        overlap_length: int = 1024,
        drop_last: bool = False,
    ):
        self.tokenizer = tokenizer
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.unicode_offset = validate_unicode_offset(unicode_offset, num_codebooks, codebook_size)
        self.sequence_length = sequence_length
        self.overlap_length = overlap_length
        self.drop_last = drop_last

        self.audio_start_token_id = None
        if audio_start_token is not None:
            self.audio_start_token_id = self.tokenizer.convert_tokens_to_ids(audio_start_token)
            if self.audio_start_token_id is None:
                raise ValueError(f"Token '{audio_start_token}' not found in tokenizer")
        self.audio_end_token_id = None
        if audio_end_token is not None:
            self.audio_end_token_id = self.tokenizer.convert_tokens_to_ids(audio_end_token)
            if self.audio_end_token_id is None:
                raise ValueError(f"Token '{audio_end_token}' not found in tokenizer")

    def _group_codes_files(self, codes_files: List[str]) -> List[List[str]]:
        grouped_codes_files = []
        last_file_root = None
        for codes_file in codes_files:
            file_root = re.match(r"(.+)_c\d+[_.]", codes_file).group(1)
            if file_root != last_file_root:
                grouped_codes_files.append([])
                last_file_root = file_root
            grouped_codes_files[-1].append(codes_file)
        return grouped_codes_files

    def iterate_examples(self, codes_path: str, codes_filter: Optional[Union[str, List[str]]] = None) -> Iterator[str]:
        codes_files = get_codes_files(codes_path, codes_filter)
        # group codes files by root filename (minus channel and starting timestamp)
        grouped_codes_files = self._group_codes_files(codes_files)
        for file_group in tqdm(grouped_codes_files, desc="Codes file groups"):
            # concatenate all codes files in each group
            codes = np.concatenate([np.load(file) for file in file_group], axis=-1)
            if len(codes.shape) == 4:
                codes = codes[0, 0]
            elif len(codes.shape) == 3:
                codes = codes[0]
            codes = codes[:self.num_codebooks]
            # convert to unicode string
            chars = codes_to_chars(
                codes, 
                self.codebook_size,
                copy_before_conversion=False,
                unicode_offset=self.unicode_offset,
            )
            # encode the unicode string with the tokenizer
            tokens = self.tokenizer.encode(chars, return_tensors="np")[0]
            sequence_length = self.sequence_length
            if self.tokenizer.bos_token_id is not None and tokens[0] == self.tokenizer.bos_token_id:
                tokens = tokens[1:]
                sequence_length -= 1
            if self.tokenizer.eos_token_id is not None and tokens[-1] == self.tokenizer.eos_token_id:
                tokens = tokens[:-1]
                sequence_length -= 1
            if self.audio_start_token_id is not None:
                sequence_length -= 1
            if self.audio_end_token_id is not None:
                sequence_length -= 1
            # yield examples from the sequence with the specified sequence length and overlap
            start = 0
            while True:
                end = start + sequence_length
                if self.drop_last and end > len(tokens):
                    break
                example_tokens = tokens[start:end]
                # add audio start and end tokens if specified
                if self.audio_start_token_id is not None:
                    example_tokens = np.concatenate([[self.audio_start_token_id], example_tokens])
                if self.audio_end_token_id is not None:
                    example_tokens = np.concatenate([example_tokens, [self.audio_end_token_id]])
                example = self.tokenizer.decode(example_tokens)
                yield example
                if end >= len(tokens):
                    break
                start = end - self.overlap_length