# Data provenance

Every dataset is pinned to a Hub commit hash in `configs/data.yaml`. `make data` downloads the pinned revisions, applies the filters below, and writes the manifest `results/data_manifest.json` with row counts before and after each filter and the SHA-256 of every output file.

## Datasets

| Use | Hub id | Revision | Split | License | Selection |
|---|---|---|---|---|---|
| Supervised fine-tuning | `HuggingFaceH4/ultrachat_200k` | `8049631c405ae6576f93f445c6b8166f76f5505a` | `train_sft` | MIT | seeded shuffle, first 20,000 rows that pass every filter |
| Preference optimization | `HuggingFaceH4/ultrafeedback_binarized` | `3949bf5f8c17c394422ccfab0c31ea9c20bdeb85` | `train_prefs` | MIT | seeded shuffle, first 20,000 rows that pass every filter |
| Distillation prompts | `trl-lib/ultrafeedback-prompt` | `2f5b1bc3e81879a720a8c870a82d668923d8aebe` | `train` | derived from `openbmb/UltraFeedback` (MIT); the dataset card states no license | seeded shuffle, first 20,000 rows that pass every filter |
| Judge prompts | `HuggingFaceH4/ultrafeedback_binarized` | `3949bf5f8c17c394422ccfab0c31ea9c20bdeb85` | `test_prefs` | MIT | 300 single-turn prompts of at most 1,024 tokens, unique by text, drawn with a fixed seed; committed as `eval/judge_prompts.jsonl` |

The preference pairs come from the public UltraFeedback binarization, not from private human labels. Token lengths are counted with the tokenizer of the 7B base model.

## Filters

1. Conversion: dialogues must alternate user and assistant turns, end with the assistant, and contain no blank turn. Preference pairs must share an identical prompt and differ in their final completion.
2. Length: the full record (prompt and completions) must fit within `max_tokens` of its dataset entry.
3. Decontamination against every evaluation text: the 300 judge prompts and the question fields of the benchmark sources below. A row is dropped on a normalized exact match or on a MinHash near-duplicate match (128 permutations, 4 rows per band, 5-word shingles, estimated Jaccard similarity of at least 0.8).
4. Uniqueness of the final user prompt after normalization.

## Benchmark sources used for decontamination

| Benchmark | Hub id | Field |
|---|---|---|
| GSM8K | `openai/gsm8k` | `question` |
| MMLU | `cais/mmlu` | `question` |
| ARC Challenge | `allenai/ai2_arc` | `question` |
| HellaSwag | `Rowan/hellaswag` | `ctx` |
| TruthfulQA | `truthfulqa/truthful_qa` | `question` |
| IFEval | `google/IFEval` | `prompt` |

Benchmark sources are loaded at their default branch head at preparation time. The manifest records the number of reference texts taken from each source.

## Models

Model revisions are pinned in `configs/models.yaml`. The 7B and 1.5B base models are released under Apache 2.0; `scripts/render_model_card.py` checks the license field before a card is published.

<!-- results:data_manifest:start -->
<!-- results:data_manifest:end -->
