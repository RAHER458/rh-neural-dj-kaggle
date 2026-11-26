# RH Neural DJ Playground — Kaggle prototype

Kaggle script kernel: https://www.kaggle.com/code/vanchesstv/rh-neural-dj-playground

## Что внутри

- `main.py` — Gradio app: upload audio files, analyze with librosa, render a real beat-aware transition with numpy, export WAV/MP3.
- `kernel-metadata.json` — Kaggle kernel metadata.

## Важно

`clawd139/automix` has no trained checkpoint and no LICENSE file; inference without weights is random.
This app does not claim to run a diffusion model. It uses a real DSP transition renderer.
