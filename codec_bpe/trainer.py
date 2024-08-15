from typing import Optional, List, Union, Iterator

import numpy as np
from tokenizers import AddedToken

from .sentencepiece_bpe import SentencePieceBPETokenizer
from .converter import codes_to_chars, UNICODE_OFFSET
from .utils import get_codes_files

class Trainer:
    def __init__(
        self, 
        num_codebooks: int,
        codebook_size: int,
        codec_framerate: Optional[int] = None,
        chunk_size_secs: Optional[int] = None,
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: Optional[List[Union[str, AddedToken]]] = None,
        unk_token: Optional[str] = None,
        max_token_codebook_ngrams: Optional[int] = None,
        unicode_offset: int = UNICODE_OFFSET,
    ):
        if chunk_size_secs is not None:
            if codec_framerate is None:
                raise ValueError("If chunk_size_secs is set, codec_framerate must also be set.")
            if chunk_size_secs < 1:
                raise ValueError("chunk_size_secs must be a positive integer >= 1.")

        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codec_framerate = codec_framerate
        self.chunk_size_secs = chunk_size_secs
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens
        self.unk_token = unk_token
        self.max_token_codebook_ngrams = max_token_codebook_ngrams
        self.unicode_offset = unicode_offset

        if self.special_tokens is None:
            self.special_tokens = []
        if self.unk_token is not None and self.unk_token not in self.special_tokens:
            self.special_tokens.insert(0, self.unk_token)

    def _iterate_and_convert(self, codes_files: List[str]) -> Iterator[str]:
        for codes_file in codes_files:
            codes = np.load(codes_file)
            if len(codes.shape) == 4:
                codes = codes[0, 0]
            elif len(codes.shape) == 3:
                codes = codes[0]
            codes = codes[:self.num_codebooks]
            chunk_size = self.chunk_size_secs * self.codec_framerate if self.chunk_size_secs else codes.shape[1]
            for i in range(0, codes.shape[1], chunk_size):
                chars = codes_to_chars(
                    codes[:, i:i+chunk_size], 
                    self.codebook_size, 
                    copy_before_conversion=False,
                    unicode_offset=self.unicode_offset
                )
                yield chars

    def train(
        self,
        codes_path: str,
        num_files: Optional[int] = None,
    ) -> SentencePieceBPETokenizer:
        # Compute base alphabet. This should be num_codebooks * codebook_size so that we never split a codeword
        # into smaller units.
        initial_alphabet = [
            chr(i) for i in range(
                self.unicode_offset, 
                self.unicode_offset + self.num_codebooks * self.codebook_size
            )
        ]
        
        # If max_token_codebook_ngrams is set, we need to limit the token length to avoid creating tokens that are larger than
        # that number of codebook ngrams. A codebook ngram is a sequence of length num_codebooks with one codeword taken from 
        # each codebook, representing a complete acoustic unit.
        # For example if num_codebooks = 4 and max_token_codebook_ngrams = 5, the maximum token length would be 20.
        max_token_length = None
        if self.max_token_codebook_ngrams is not None and self.max_token_codebook_ngrams > 0:
            max_token_length = self.max_token_codebook_ngrams * self.num_codebooks

        # Train tokenizer
        codes_files = get_codes_files(codes_path, num_files)
        codes_iterator = self._iterate_and_convert(codes_files)
                
        tokenizer = SentencePieceBPETokenizer(unk_token=self.unk_token, add_prefix_space=False)
        tokenizer.train_from_iterator(
            codes_iterator,
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            limit_alphabet=len(initial_alphabet),
            initial_alphabet=initial_alphabet,
            max_token_length=max_token_length,
        )
        return tokenizer
    