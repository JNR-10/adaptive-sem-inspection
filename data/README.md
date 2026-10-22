# Data directory

This directory deliberately contains no raw images or archives.

Use a runtime directory outside the repository, for example
`/fast-ssd/adaptive-sem-runtime`, and pass it through `--runtime-root`. The
native pipeline creates its own `data/raw`, `data/extracted`, and
`data/manifests` directories there.

MIIC is CC BY-NC 4.0 and is not redistributed with this project. See
`docs/native_gpu.md` and `docs/data_card.md` before acquiring it.
