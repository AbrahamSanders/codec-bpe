# codec-bpe
Implementation of [Acoustic BPE](https://arxiv.org/abs/2310.14580) (Shen et al., 2024), extended for RVQ-based Neural Audio Codecs such as [EnCodec](https://github.com/facebookresearch/encodec) (Défossez et al., 2022) or [DAC](https://github.com/descriptinc/descript-audio-codec) (Kumar et al., 2023). Built on top of the [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) library.

## Setup
Clone the repository and run
```bash
pip install -r requirements.txt
```
A self-contained PyPI package is coming soon!

## Usage

### Convert audio codes to and from unicode strings
Use your codec of choice (e.g., EnCodec, DAC) to encode your audio into a torch tensor or numpy array of codes of shape (num_codebooks, length), then use the provided converter methods to convert to and from unicode strings.

**Note:** In the Acoustic BPE paper, a single-level codec was used (HuBERT + k-means), where each encoded timestep consisted of a single code which was converted to a single unicode character. Here, we support multi-level codecs based on Residual Vector Quantizers. If num_codebooks > 1, a flattening pattern is used to interleave all codebooks into a single level before mapping to unicode. For example, if 4 codebooks are used then each encoded timestep would consist of 4 codes (one from each codebook) and would be converted to a unicode 4-gram.

For example, using EnCodec 24 kHz at 3 kbps (4 codebooks):
```python
import torch
import librosa
from transformers import EncodecModel, AutoProcessor
from codec_bpe import codes_to_chars, chars_to_codes

# encode audio using EnCodec
encodec_model = "facebook/encodec_24khz"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = EncodecModel.from_pretrained(encodec_model).to(device)
processor = AutoProcessor.from_pretrained(encodec_model)

audio, sr = librosa.load("some_audio.mp3", sr=model.config.sampling_rate, mono=True)
inputs = processor(raw_audio=audio, sampling_rate=sr, return_tensors="pt").to(device)
with torch.no_grad():
    encoded_audio = model.encode(**inputs, bandwidth=3.0).audio_codes[0, 0]

# convert codes to unicode string
unicode_str = codes_to_chars(encoded_audio, codebook_size=model.codebook_size)

# restore unicode string to codes tensor
encoded_audio_2 = chars_to_codes(
    unicode_str, 
    num_codebooks=encoded_audio.shape[0], 
    codebook_size=model.codebook_size, 
    return_tensors="pt"
)

assert torch.equal(encoded_audio.cpu(), encoded_audio_2)
```

### Train a tokenizer from audio files
To train a tokenizer from audio files:

1. Use your codec of choice (e.g., EnCodec, DAC) to encode each audio file into a directory of numpy arrays (.npy files) of the shape (num_codebooks, length). Also acceptable is (1, num_codebooks, length) or (1, 1, num_codebooks, length). A utility to do this for you is coming soon!

2. Suppose your directory of audio codes numpy arrays is called `codes` and you want to use the first 4 codebooks of [EnCodec 24 kHz](https://huggingface.co/facebook/encodec_24khz), run:
```bash
python train_tokenizer.py \
    --codes_path codes \
    --num_codebooks 4 \
    --codebook_size 1024 \
    --codec_framerate 75 \
    --chunk_size_secs 30 \
    --vocab_size 30000 \
    --save_path output/tokenizer.json
```
Here: 
- `num_codebooks` specifies how many codebooks should be used (in a flattened pattern) when converting each timestep to unicode. For example, EnCodec 24kHz uses 2 codebooks at 1.5 kbps, 4 codebooks at 3 kbps, 8 codebooks at 6 kbps, etc.
- `codebook_size` specifies the size of the codebook. EnCodec 24 kHz uses a codebook size of 1024.
- `codec_framerate` specifies the framerate (number of timesteps per second) of the codec. EnCodec 24 kHz generates 75 timesteps per second.
- `chunk_size_secs` specifies the number of timesteps (in seconds) that get converted to unicode and returned to the underlying Tokenizers trainer at a time.
- `vocab_size` specifies the number of tokens (including the base vocabulary of individual unicode characters) that you want your tokenizer to have. The base vocabulary size is `num_codebooks` x `codebook_size`. For example, the command above would yield a tokenizer with a base vocabulary of 4096 individual unicode character tokens, each representing a single code from a single codebook, and 25,904 merged "ngram" tokens.

See [train_tokenizer.py](train_tokenizer.py) for a complete list of supported arguments.

### Extend an existing Transformers PreTrainedTokenizer
You may want to train a new codec BPE tokenizer and then export its trained vocabulary to an existing Transformers tokenizer. For example, extending the Llama3, Mistral, Qwen, etc. tokenizers for multimodal text-audio language modeling.

Suppose you have trained your codec BPE tokenizer and saved it to `output/tokenizer.json` and you want to extend the Mistral-7B-v0.1 tokenizer with its vocabulary, run:
```bash
python extend_tokenizer.py \
    --transformers_tokenizer mistralai/Mistral-7B-v0.1 \
    --codec_bpe_tokenizer output/tokenizer.json \
    --audio_start_token <audio> \ # optional
    --audio_end_token </audio>    # optional
```
This will simply add every token in `output/tokenizer.json` to the `mistralai/Mistral-7B-v0.1` tokenizer as a special token and save a copy of the latter. 

#### Avoiding vocabulary conflicts
If the added codec BPE unicode tokens would conflict with existing tokens in the vocabulary, there are two options to mitigate this:

1. Override the default unicode offset using the `unicode_offset` argument for both `train_tokenizer.py` and `extend_tokenizer.py`. By default, unicode characters from the [CJK Unified Ideographs](https://symbl.cc/en/unicode-table/#cjk-unified-ideographs) block are used, following the Acoustic BPE paper. You can set `unicode_offset` to a different value to use a different unicode block that doesn't conflict with your existing vocabulary.

2. Use the `use_special_token_format` argument for `extend_tokenizer.py`. This wraps each unicode character in each ngram with <>. For example, the 4-gram token "一刁嘂娃" would be converted to a token containing the string "\<一>\<刁>\<嘂>\<娃>". This format is more verbose, but should virtually eliminate the possibility of a vocabulary conflict:
    ```bash
    python extend_tokenizer.py \
        --transformers_tokenizer mistralai/Mistral-7B-v0.1 \
        --codec_bpe_tokenizer output/tokenizer.json \
        --audio_start_token <audio> \ # optional
        --audio_end_token </audio> \  # optional
        --use_special_token_format
    ```
    Then when preparing audio for tokenization with the extended tokenizer, you can pass the same argument to the `codes_to_chars` function:
    ```python
    # convert codes to unicode string
    unicode_str = codes_to_chars(encoded_audio, codebook_size=model.codebook_size, use_special_token_format=True)
    ```
    It is unnecessary to pass this argument to `chars_to_codes` - it will automatically detect and remove the special token format before converting back to codes.