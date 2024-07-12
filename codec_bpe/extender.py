from typing import Optional
from transformers import PreTrainedTokenizer
from tokenizers import Tokenizer
from tqdm import trange

from .converter import UNICODE_OFFSET

def extend_transformers_tokenizer(
    transformers_tokenizer: PreTrainedTokenizer,
    codec_bpe_tokenizer: Tokenizer,
    num_codebooks: Optional[int] = None,
    codebook_size: Optional[int] = None,
    use_special_code_format: bool = False,
    unicode_offset: int = UNICODE_OFFSET,
) -> int:
    if use_special_code_format:
        if num_codebooks is None or codebook_size is None:
            raise ValueError("num_codebooks and codebook_size must be specified when using the special code format.")
        conversion_dict = {}
        for i in range(num_codebooks):
            for j in range(codebook_size):
                code_char = chr(unicode_offset + i*codebook_size + j)
                special_token = f"<c{i}t{j:04d}>"
                conversion_dict[code_char] = special_token

    vocab_size = codec_bpe_tokenizer.get_vocab_size()
    target_tokens = []
    for i in trange(vocab_size):
        token = codec_bpe_tokenizer.id_to_token(i)
        if use_special_code_format and token[0] in conversion_dict:
            token = "".join([conversion_dict[c] for c in token])
        target_tokens.append(token)

    return transformers_tokenizer.add_tokens(target_tokens, special_tokens=True)
