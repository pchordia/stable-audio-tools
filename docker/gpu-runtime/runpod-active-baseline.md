# Runpod Active Baseline

Captured from the active Runpod training pod `xmplo9ffm6ut2i` on March 15, 2026.

Host baseline:

- GPU: `NVIDIA H200`
- Driver: `570.211.01`
- Python: `3.11.10`
- Kernel: `6.8.0-94-generic`
- FFmpeg: `4.4.2-0ubuntu0.22.04.1`

Verified runtime state:

- `flash_attn` import works
- `stable_audio_tools.models.transformer.flash_attn_func is not None`

Pinned package versions observed:

- `torch==2.10.0`
- `torchaudio==2.10.0`
- `flash_attn==2.7.3`
- `torchcodec==0.10.0`
- `soundfile==0.13.1`
- `pytorch-lightning==2.1.0`
- `wandb==0.25.0`
- `einops==0.8.2`
- `numpy==1.23.5`
- `fsspec==2026.2.0`
- `transformers==5.2.0`

This file is the provenance record for the Nebius GPU runtime baseline.
