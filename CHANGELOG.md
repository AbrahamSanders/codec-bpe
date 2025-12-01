**2025-12-01**
- Added support for [NeuCodec](https://huggingface.co/neuphonic/neucodec), a new high-quality single-level codec with a 50 Hz framerate! NeuCodec extends XCodec2 with inference speedups and a commercially permissive license. Use `--codec_model neuphonic/neucodec` when encoding audio with `codec_bpe.audio_to_codes` to encode using the NeuCodec model. See [here](README.md#train-a-tokenizer-from-audio-files) for a usage example.

**2025-06-22**
- Added support for [MagiCodec](https://github.com/Ereboas/MagiCodec), a new **streaming** single-level codec with a 50 Hz framerate! Use `--codec_model MagiCodec-50Hz-Base` when encoding audio with `codec_bpe.audio_to_codes` to encode using the MagiCodec model. See [here](README.md#train-a-tokenizer-from-audio-files) for a usage example.

**2025-06-19**
- Added ability to encode audio into subsecond chunk sizes with a sliding window of prior audio as context. This helps support use-cases where the encoded audio should simulate a streaming setting. For example, many codecs will encode the same audio differently depending on the encoder's receptive field size - even with native streaming codecs like Mimi. So, when training a streaming speech-to-text audio LM, we want to encode the training audio in tiny chunks so that it resembles what will be received during live streaming. This helps prevent throwing the model out of distribution at inference time.
  - Use the `--chunk_size_secs` and `--context_secs` parameters with `codec_bpe.audio_to_codes` to configure this.
  - By default `--chunk_size_secs=30` and `--context_secs=0.0` for non-streaming usage. 
  - `--context_secs` controls the sliding window encoding size, which is useful to avoid codec degradation at tiny chunk sizes. For example, `--chunk_size_secs=0.08` with `--context_secs=0.4` will encode audio in chunks of 80ms, each chunk receiving the previous 320ms of audio as context to the encoder's receptive field (we encode 320 + 80 = 400ms of audio at a time but only keep the final 80ms of codes).

**2025-06-16**
- Added support for [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) and [SimVQ](https://github.com/youngsheen/SimVQ)! Both are single-level codecs that share the same architecture but differ in their VQ strategy. WavTokenizer comes in 40Hz and 75Hz variants with a vocabulary size of 4096. SimVQ variants have a 75Hz framerate with vocabulary sizes ranging from 4096 to 262144 codes. SimVQ also features a causal encoder and partially causal decoder, making it suitable for streaming use cases. 
  - Use `--codec_model WavTokenizer-large-320-24k-4096` (or any other from the `Model` column on [this table](#supported-codecs)) with `codec_bpe.audio_to_codes` to encode audio using WavTokenizer.
  - Use `--codec_model simvq_4k` (or any other from the `Model` column on [this table](#supported-codecs)) with `codec_bpe.audio_to_codes` to encode audio using SimVQ.
  - See [here](README.md#train-a-tokenizer-from-audio-files) for usage examples.

**2025-04-07**
- Added support for [XCodec2](https://huggingface.co/HKUSTAudio/xcodec2), a high-quality multilingual single-level codec with a 50 Hz framerate! Use `--codec_model HKUSTAudio/xcodec2` when encoding audio with `codec_bpe.audio_to_codes` to encode using the XCodec2 model. See [here](README.md#train-a-tokenizer-from-audio-files) for a usage example.

**2025-03-09**
- Added support for [FunCodec](https://funcodec.github.io/) from Alibaba DAMO Speech Lab! Use `--codec_model alibaba-damo/...` when encoding audio with `codec_bpe.audio_to_codes` to encode using the FunCodec model. Model paths on the HuggingFace hub are listed [here](https://github.com/modelscope/FunCodec?tab=readme-ov-file#available-models). See [here](README.md#train-a-tokenizer-from-audio-files) for a usage example.

**2024-09-20**
- Added support for Kyutai Lab's [Mimi codec](https://huggingface.co/kyutai/mimi), an amazing new codec with a 12.5 Hz framerate! Use `--codec_model kyutai/mimi` when encoding audio with `codec_bpe.audio_to_codes` to encode using the Mimi model. See [here](README.md#train-a-tokenizer-from-audio-files) for a usage example.

**2024-09-19**
- Initial Release!