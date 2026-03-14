# CPU Transcode Container

This image is the lightweight runtime for the Nebius CPU transcode stage.

It is intentionally small and only contains what the chunking pipeline needs:

- Python 3.11
- `ffmpeg`
- `boto3`

The transcode script itself is mounted from the deploy workspace on the VM, so the image does not need to vendor the repo contents.

## Build

```bash
docker buildx build --platform linux/amd64 \
  -f docker/cpu-transcode/Dockerfile \
  -t stable-audio-tools:cpu-transcode \
  --load .
```

## Notes

- This image is for CPU-only staging and chunking.
- The default published tag is intended to be `ghcr.io/pchordia/stable-audio-tools:cpu-transcode`.
