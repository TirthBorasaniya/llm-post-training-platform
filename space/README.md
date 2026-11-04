---
title: Distilled versus base student
sdk: gradio
sdk_version: 6.31.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# Distilled student versus base student

A side-by-side comparison of a Qwen 2.5 1.5B Instruct model distilled from an aligned 7B teacher and the original 1.5B model. Both run in Q4_K_M quantization on the Space's CPU through `llama-cpp-python`.

## Configuration

Set these variables in the Space settings:

| Variable | Meaning |
|---|---|
| `GGUF_REPO` | Hub repository holding the GGUF files, in the form `owner/name` |
| `DISTILLED_FILE` | file name of the distilled student, default `distilled_1_5b-Q4_K_M.gguf` |
| `BASE_FILE` | file name of the base student, default `base_1_5b-Q4_K_M.gguf` |
| `RESULTS_URL` | optional link to the evaluation results |

Each session is limited to six requests per minute and 256 new tokens per answer.
The requirements file adds the prebuilt CPU wheel index of `llama-cpp-python`, so the build does not compile it from source.
A Space on the free tier sleeps when idle.
