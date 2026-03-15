# GPU Runtime Container

This container is the first pinned GPU runtime for Nebius/Runpod encode and training jobs.

It is intended to replace bootstrap-heavy VM setup with a reproducible runtime that already contains:

- the pinned `stable-audio-tools` fork,
- `ffmpeg`,
- `torchcodec`,
- `soundfile`,
- `flash-attn`,
- a simple validation step that fails the build if `flash_attn` does not import.

Current baseline source:

- `/Users/paragchordia/.codex/workspaces/stable-audio-tools/docker/gpu-runtime/runpod-active-baseline.md`

The Dockerfile now intentionally tracks the active Runpod training environment:

- base image: `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel`
- `flash-attn==2.7.3`
- additional pinned runtime packages in `docker/gpu-runtime/runpod-baseline-requirements.txt`

## Build

```bash
docker buildx build --platform linux/amd64 \
  -f docker/gpu-runtime/Dockerfile \
  -t stable-audio-tools:gpu-runtime \
  --load .
```

## Validate

```bash
docker run --rm --gpus all stable-audio-tools:gpu-runtime python scripts/validate_flash_attn.py
```

## Notes

- This container assumes a host with a working NVIDIA driver stack.
- It does not replace the need for a GPU-capable host image.
- For future H100/H200 runs, this image should be treated as the default runtime baseline rather than reinstalling Python and `flash-attn` during VM bootstrap.
- On Apple Silicon, build for `linux/amd64` explicitly. The Docker build already runs `python scripts/validate_flash_attn.py`, so that build step is the reliable local validation signal even if `docker run` on the final amd64 image is not usable on the laptop.
