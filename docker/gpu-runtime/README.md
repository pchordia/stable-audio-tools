# GPU Runtime Container

This container is the first pinned GPU runtime for Nebius/Runpod encode and training jobs.

It is intended to replace bootstrap-heavy VM setup with a reproducible runtime that already contains:

- the pinned `stable-audio-tools` fork,
- `ffmpeg`,
- `torchcodec`,
- `soundfile`,
- `flash-attn`,
- a simple validation step that fails the build if `flash_attn` does not import.

## Build

```bash
docker build -f docker/gpu-runtime/Dockerfile -t stable-audio-tools:gpu-runtime .
```

## Validate

```bash
docker run --rm --gpus all stable-audio-tools:gpu-runtime python scripts/validate_flash_attn.py
```

## Notes

- This container assumes a host with a working NVIDIA driver stack.
- It does not replace the need for a GPU-capable host image.
- For future H100/H200 runs, this image should be treated as the default runtime baseline rather than reinstalling Python and `flash-attn` during VM bootstrap.
