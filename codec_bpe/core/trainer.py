from typing import Optional, List, Union, Iterator
import warnings
import numpy as np
from tokenizers import AddedToken
from transformers import PreTrainedTokenizerFast

from .sentencepiece_bpe import SentencePieceBPETokenizer
from .converter import codes_to_chars, validate_unicode_offset, UNICODE_OFFSET
from .utils import get_codes_files

class Trainer:
    def __init__(
        self, 
        num_codebooks: int,
        codebook_size: int,
        codec_framerate: Optional[float] = None,
        chunk_size_secs: Optional[int] = None,
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: Optional[List[Union[str, AddedToken]]] = None,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        unk_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        max_token_codebook_ngrams: Optional[int] = None,
        unicode_offset: int = UNICODE_OFFSET,
    ):
        if chunk_size_secs is not None:
            if codec_framerate is None:
                raise ValueError("If chunk_size_secs is set, codec_framerate must also be set.")
            if chunk_size_secs < 1:
                raise ValueError("chunk_size_secs must be a positive integer >= 1.")
        if eos_token is None and pad_token is None:
            raise ValueError(
                "Either pad_token or eos_token should be set, otherwise padded batching will not work with this tokenizer."
            )
        if max_token_codebook_ngrams is not None and max_token_codebook_ngrams < 0:
            raise ValueError("max_token_codebook_ngrams must be a non-negative integer (0 or greater).")

        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codec_framerate = codec_framerate
        self.chunk_size_secs = chunk_size_secs
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.max_token_codebook_ngrams = max_token_codebook_ngrams
        self.unicode_offset = validate_unicode_offset(unicode_offset, num_codebooks, codebook_size)

        if self.special_tokens is None:
            self.special_tokens = []
        for special_token in [self.eos_token, self.bos_token, self.unk_token, self.pad_token]:
            if special_token is not None and special_token not in self.special_tokens:
                self.special_tokens.insert(0, special_token)

        min_vocab_size = self.num_codebooks*self.codebook_size + len(self.special_tokens)
        if self.vocab_size < min_vocab_size:
            raise ValueError(
                f"vocab_size is set to {self.vocab_size} but it must be at least {min_vocab_size} to accommodate "
                f"{self.num_codebooks} x {self.codebook_size} codes and {len(self.special_tokens)} special token(s).\n"
                f"Consider setting vocab_size to {min_vocab_size} + K, where K is the number of tokens you want to "
                "reserve for codebook ngrams (learned merges). K should be a sufficiently large number (e.g. >= 10,000) "
                "to allow for wide coverage of the most common codebook ngrams in your training data."
            )

    def _iterate_and_convert(self, codes_files: List[str]) -> Iterator[str]:
        for codes_file in codes_files:
            codes = np.load(codes_file)
            if len(codes.shape) == 4:
                codes = codes[0, 0]
            elif len(codes.shape) == 3:
                codes = codes[0]
            codes = codes[:self.num_codebooks]
            chunk_size = int(self.chunk_size_secs * self.codec_framerate) if self.chunk_size_secs else codes.shape[1]
            for i in range(0, codes.shape[1], chunk_size):
                chars = codes_to_chars(
                    codes[:, i:i+chunk_size], 
                    self.codebook_size, 
                    copy_before_conversion=False,
                    unicode_offset=self.unicode_offset,
                )
                yield chars

    def train(
        self,
        codes_path: str,
        codes_filter: Optional[Union[str, List[str]]] = None, 
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
        if self.max_token_codebook_ngrams is not None:
            max_token_length = max(1, self.max_token_codebook_ngrams * self.num_codebooks)

        # Train tokenizer
        if max_token_length == 1:
            # We don't need to actually train the tokenizer here, just create one with the initial alphabet.
            codes_iterator = []
        else:
            codes_files = get_codes_files(codes_path, codes_filter, num_files)
            if not self.chunk_size_secs and codes_files[0].split("_")[-1].startswith("c"):
                warnings.warn(
                    "The codes files do not have start timestamps, indicating they represent full-length encoded audio files rather than chunks. "
                    "It is recommended to set `--chunk_size_secs` to a small value (e.g. 30) to avoid the tokenizer training on very long sequences. "
                    "Training on very long sequences of audio codes can lead to memory issues and poor BPE merges."
                )
            codes_iterator = self._iterate_and_convert(codes_files)
            # the +1 is because max_token_length is exclusive (e.g., max_token_length of n yields an actual max token length of n-1).
            # not sure if this is a bug in Tokenizers or intended behavior.
            max_token_length = max_token_length + 1 if max_token_length is not None else None

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
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, 
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            clean_up_tokenization_spaces=False,
            model_input_names=['input_ids', 'attention_mask'],
        )
        return tokenizer
    