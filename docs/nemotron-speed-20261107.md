# Nemotron 3.5 dictation speed - November 7, 2026

Context: after switching Fn dictation to Nemotron 3.5 Transcribe 4-bit, a
33-second dictation took 5-8 seconds. Measured on the M4 Max, with Low Power
Mode on while plugged in (`pmset -g custom`: AC `powermode 1`), and other work
using the CPU and about a quarter of the GPU.

## Where the time goes

Encoding the audio takes about 0.1-0.5 s. The language model takes the rest:

| Stage | 2-s clip | 33-s clip |
|---|---:|---:|
| Prompt + audio prefill | 0.8-1.1 s (92 tokens) | 2.3-2.7 s (474 tokens) |
| Text decode | 0.5-0.6 s (8 tokens) | 5-9 s (66 tokens) |

Decode ran at 7-20 tokens/s, where this chip should manage roughly 60-100.
Building each token's graph takes about 6 ms on the CPU; GPU execution takes 43-68 ms.
Raw GPU checks in the same state measured about 90 GB/s read bandwidth and
1.6-2.2 TFLOPS bf16, roughly a fifth of an unthrottled M4 Max. Decode is
bandwidth-bound (about 2.5 GB of active 4-bit weights per token), so Low Power
Mode is the biggest single factor. It is a system setting, not something the app changes.

## Change kept: reuse the prompt prefix

Nemotron's ASR prompt places the audio after 56 constant tokens (empty system
turn plus the transcription instruction). `NemotronPrefixCache` in
`mlx_next_asr.py` computes their attention/Mamba state once at load. Each request
then prefills only its audio plus the 8-token assistant suffix.

- Interleaved medians, Low Power Mode: 0.45-0.6 s saved on 4-16 s clips, about
  20%; noisy at 2 s and 24-33 s because of background GPU load.
- Text matched the previous path exactly on 12 of 13 clips, including the
  163-second chunked recording and Zack's 33-second dictation, plus every repeated
  benchmark run. One 15 dB noise clip changed at a near-tie
  ("mantelboard" became "mantel board" and the first word became lowercase). That
  happens because the prefix is now computed in its own pass. That clip went from 2
  normalized word errors to 0.

## Rejected: MTP speculative decoding

The community 4-bit export omits NVIDIA's MTP draft head. The head from
`nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4` (`model-mtp-fp8.safetensors`,
1.34 GB) was dequantized to BF16 and grafted onto the 4-bit export. Output stayed
identical, but it was slower: about 1.9 tokens per verification step, because
the second and later drafts were almost never accepted. Each multi-token
verification also cost about 1.65x a single step, because MoE verification loads more experts. MLX Audio's
own speculative path took 10-11 s versus 7.2 s without it on the 33-second clip.
The head was probably trained with NVIDIA's release backbone, not the early-access
weights behind the community export. The grafted files were deleted.

## Not tried

- Turning off Low Power Mode on AC; that is the owner's system setting. The battery
  showed "not charging" at 34% on a 96 W adapter even in Low Power Mode, so heavy
  background work may drain it with Low Power Mode off.
- Transcribing pause-separated chunks while recording. This could hide most of
  the latency of long dictations, but each chunk would lose surrounding context.
  It needs an accuracy comparison on real dictation first.
