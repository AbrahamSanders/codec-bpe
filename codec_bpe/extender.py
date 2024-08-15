from typing import Optional, Union
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tqdm import trange

def extend_transformers_tokenizer(
    transformers_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    codec_bpe_tokenizer: Tokenizer,
    audio_start_token: Optional[str] = None,
    audio_end_token: Optional[str] = None,
    use_special_token_format: bool = False,
) -> int:
    additional_special_tokens = []
    if audio_start_token is not None:
        additional_special_tokens.append(audio_start_token)
    if audio_end_token is not None:
        additional_special_tokens.append(audio_end_token)
    vocab_size = codec_bpe_tokenizer.get_vocab_size()
    target_tokens = []
    for i in trange(vocab_size):
        token = codec_bpe_tokenizer.id_to_token(i)
        if use_special_token_format:
            token = "".join([f"<{c}>" for c in token])
        target_tokens.append(token)

    num_added = 0
    if len(additional_special_tokens) > 0:
        num_added += transformers_tokenizer.add_special_tokens(
            special_tokens_dict={"additional_special_tokens": additional_special_tokens}, 
            replace_additional_special_tokens=False,
        )
    num_added += transformers_tokenizer.add_tokens(target_tokens, special_tokens=True)
    return num_added
