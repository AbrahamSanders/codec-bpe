from typing import Optional, Union, List
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from tqdm import trange

def extend_existing_tokenizer(
    existing_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    codec_bpe_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    additional_special_tokens: Optional[List[str]] = None,
) -> int:
    target_tokens = []
    skip_token_ids = set([
        codec_bpe_tokenizer.bos_token_id, 
        codec_bpe_tokenizer.eos_token_id, 
        codec_bpe_tokenizer.unk_token_id, 
        codec_bpe_tokenizer.pad_token_id,
    ])
    for i in trange(len(codec_bpe_tokenizer)):
        if i in skip_token_ids:
            continue
        token = codec_bpe_tokenizer.convert_ids_to_tokens(i)
        target_tokens.append(token)

    num_added = 0
    if additional_special_tokens:
        num_added += existing_tokenizer.add_special_tokens(
            special_tokens_dict={"additional_special_tokens": additional_special_tokens}, 
            replace_additional_special_tokens=False,
        )
    num_added += existing_tokenizer.add_tokens(target_tokens, special_tokens=True)
    return num_added
