# LLM Output Diversity Pipeline (Creative Writing Bench)

This repo now contains an end-to-end pipeline to:
- Query multiple LLMs on Creative Writing Bench prompts with appended `seed_modifiers`
- Sample each prompt condition 8 times per model
- Save raw outputs in per-model JSON files
- Analyze diversity in embedding space (`text-embedding-3-large`)
- Build placeholder artifacts for LLM-as-a-judge evaluation

## Models currently configured
- `gemini-3-flash-preview`
- `gemini-3-pro-preview`
- `GPT-5.2 pro (reasoning low)`
- `GPT-5.2 pro (reasoning high)`
- `claude-opus-4-6 (thinking disabled)`
- `claude-opus-4-6 (thinking enabled)`
- `grok-4-1-fast-non-reasoning`
- `grok-4-1-fast-reasoning`

No Qwen models are included for now.

## Prompt subset
The pipeline requests prompts `27..33` inclusive. Your dataset currently contains `28..33` in that region, so `27` is logged as missing and skipped.

Each selected prompt uses all listed `seed_modifiers` (unless you cap via CLI), and each `(prompt, seed_modifier)` condition is sampled `8` times.

## Setup
1. Create a virtual env and install deps:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Create `.env` from example:
```bash
cp .env.example .env
```

3. Fill `.env` with keys:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `XAI_API_KEY`

## Run
### 1) Generate samples
```bash
python pipeline.py generate
```

Optional knobs:
```bash
python pipeline.py generate --temperature 1.0 --top-p 1.0 --max-tokens 1100 --max-modifiers 2
```
You can parallelize calls per model:
```bash
python pipeline.py generate --max-workers 8
```

### 2) Embedding diversity analysis
Uses OpenAI embeddings model `text-embedding-3-large` by default.
```bash
python pipeline.py analyze-embeddings
```

### 3) LLM-as-judge placeholders
Builds judge prompts + placeholder result fields (no judge API calls yet).
```bash
python pipeline.py build-judge-placeholders
```

## Output structure
- `outputs/generations/*.json`: one file per model with all generation records
- `outputs/analysis/embedding_diversity.json`: pairwise cosine-distance metrics and model ranking
- `outputs/judge/judge_placeholder_*.json`: judge prompt templates + `judge_result: null`

## Generation JSON format (per model)
Top-level fields:
- `run_id`, `created_at_utc`
- `model`: provider + display name + API model id + thinking label
- `dataset`: prompt ids requested and missing ids
- `generation_config`
- `records`: flat list of generation attempts

Each record includes:
- `record_id`
- `prompt_id`, `prompt_title`, `prompt_category`
- `seed_modifier_index`, `seed_modifier`
- `sample_index` (0..7)
- `base_prompt`, `final_prompt` (seed modifier appended)
- `response_text`
- `latency_sec`
- `raw_api_response`
- `error` (null on success)

## Notes / caveats
- The pipeline uses direct HTTPS requests (`urllib.request`) for all providers, including Anthropic `POST /v1/messages` in the style you requested.
- Some model IDs as written may need provider-specific normalization in practice (for example, spacing/punctuation differences in API model names). If a model call fails, update `MODEL_SPECS` in `pipeline.py`.
- The `thinking` vs `non_reasoning` label is tracked in metadata. For OpenAI reasoning variants, the code additionally attaches `{"reasoning": {"effort": "high"}}`.
- Judge step is intentionally a placeholder scaffold for now; it writes the exact prompt and storage format for later API integration.

## Main file
- `pipeline.py`: generation + embedding analysis + judge-placeholder generation
