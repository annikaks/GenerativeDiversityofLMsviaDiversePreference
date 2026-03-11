# LLM Diversity + Accuracy Evaluation

This repository currently supports two concrete things:

1. Baseline generation for prompt-conditioned creative writing outputs.
2. Downstream evaluation of those outputs with:
   - embedding-based diversity metrics
   - LLM-as-judge prompt-adherence accuracy

The repository does **not** yet implement DivPO fine-tuning. The current codebase is set up to produce the baseline data and evaluation artifacts that you will use before and after post-training.

## Current Recommendation

If your actual goal is DivPO on post-trainable models, use the open-source path, not the old API-baseline path.

Recommended baseline family:
- `Qwen/Qwen3-8B` with thinking disabled
- `Qwen/Qwen3-8B` with thinking enabled
- `meta-llama/Llama-3.1-8B-Instruct` as an additional open baseline if your Hugging Face account has gated-model access

The open-source generator writes the **same JSON schema** as the older API pipeline, so the analysis and judge scripts still work.

## Environment

Use a Python 3.10+ environment. Python 3.8 is too old for the current requirements.

Example with conda:

```bash
conda create -n divpo python=3.11
conda activate divpo
python -m pip install -r requirements.txt
```

If you plan to run on Modal, install the Modal CLI too:

```bash
python -m pip install modal
modal setup
```

If you use API-based judging, create `.env` and set the relevant keys:

```bash
cp .env.example .env
```

Possible keys:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `XAI_API_KEY`

## Dataset Split

Current prompt usage in code:
- training / future post-training set: prompts `1..26`
- test / baseline evaluation set: prompts `28..33`
- prompt `27` is missing in the dataset and is skipped automatically

Each selected prompt uses all available `seed_modifiers` by default.

## Baseline Pipeline

### Step 1: Generate open-source baseline outputs

Run locally:

```bash
python generate_open_source.py --preset tinker-baselines
```

What this runs:
- `Qwen/Qwen3-8B` as `qwen3-8b-non-reasoning`
- `Qwen/Qwen3-8B` as `qwen3-8b-reasoning`
- `meta-llama/Llama-3.1-8B-Instruct` as `llama-3-1-8b-instruct`

What it does:
- loads prompts `28..33`
- uses `11` prompt variants per prompt by default:
  - the unmodified base prompt
  - all `10` seed-modified prompts
- generates `8` responses per prompt-condition by default
- batches generations locally with `--batch-size 4` by default
- writes one file per model to `outputs/generations/`

Explicit version:

```bash
python generate_open_source.py --preset tinker-baselines --max-modifiers 10 --num-samples 8 --batch-size 4
```

Expected outputs:
- `outputs/generations/qwen3-8b-non-reasoning.json`
- `outputs/generations/qwen3-8b-reasoning.json`
- `outputs/generations/llama-3-1-8b-instruct.json`

### Step 1b: Generate on Modal

If local generation is too slow, use Modal for the baseline generator. The Modal entrypoint:
- runs the same `generate_open_source.py` script
- mounts persistent volumes for outputs and HF cache
- supports detached runs
- resumes from the existing JSON if you restart the same model

Files used:
- [modal_generate_open_source.py](/Users/annikaks/Desktop/_stanford/cs224N-NLPwDL/project_LMDiversity/modal_generate_open_source.py)
- [generate_open_source.py](/Users/annikaks/Desktop/_stanford/cs224N-NLPwDL/project_LMDiversity/generate_open_source.py)

Recommended first run, Qwen non-reasoning only:

```bash
modal run -d modal_generate_open_source.py \
  --model-spec 'qwen3-8b-non-reasoning|Qwen/Qwen3-8B|non_reasoning' \
  --max-modifiers 10 \
  --num-samples 8 \
  --batch-size 4 \
  --max-tokens 700
```

Then Qwen reasoning:

```bash
modal run -d modal_generate_open_source.py \
  --model-spec 'qwen3-8b-reasoning|Qwen/Qwen3-8B|reasoning' \
  --max-modifiers 10 \
  --num-samples 8 \
  --batch-size 4 \
  --max-tokens 700
```

If you have gated-model access, Llama:

```bash
modal run -d modal_generate_open_source.py \
  --model-spec 'llama-3-1-8b-instruct|meta-llama/Llama-3.1-8B-Instruct|non_reasoning' \
  --max-modifiers 10 \
  --num-samples 8 \
  --batch-size 4 \
  --max-tokens 700
```

How Modal storage is organized:
- generated JSONs are written under `/vol/outputs/generations/` inside Modal
- HF model downloads are cached under `/vol/hf-cache/`
- both live on persistent Modal volumes

How to inspect or rerun:
- use detached mode (`-d`) for overnight runs
- rerun the same command to resume; `generate_open_source.py` skips records already present in the output JSON

Optional private repo sync:
- if you want the container to pull the latest private GitHub repo on startup, create a Modal secret with `GITHUB_TOKEN` and set `GITHUB_REPO_URL`
- `modal_generate_open_source.py` will clone on first run and `git pull --ff-only` on later runs
- if those env vars are absent, Modal uses the local source tree bundled into the image instead

### Step 2: Compute embedding-space diversity

Run:

```bash
python pipeline.py analyze-embeddings
python pipeline.py analyze-baseline-deviation
```

What these do:

`analyze-embeddings`
- computes per-group embedding distances
- caches embeddings in `outputs/embeddings/`
- writes per-model metrics to `outputs/analysis/*_embedding_metrics.json`

`analyze-baseline-deviation`
- computes the two main baseline metrics you have been using:
  - average cosine deviation to centroid
  - max cosine deviation to centroid
- writes per-model metrics and details files

Expected outputs:
- `outputs/embeddings/*_embeddings.json`
- `outputs/analysis/*_embedding_metrics.json`
- `outputs/analysis/*_baseline_deviation_metrics.json`
- `outputs/analysis/*_baseline_deviation_details.json`

### Step 3: Compute prompt-adherence accuracy

Recommended run:

```bash
python run_accuracy_batched.py \
  --judge-provider anthropic \
  --judge-model claude-opus-4-6 \
  --batch-size 4 \
  --generation-glob '*qwen3*.json'
```

And for Llama:

```bash
python run_accuracy_batched.py \
  --judge-provider anthropic \
  --judge-model claude-opus-4-6 \
  --batch-size 4 \
  --generation-glob '*llama-3-1-8b-instruct*.json'
```

What this does:
- reads only matching generation files
- scores prompt adherence with the specified judge model
- writes one accuracy file per model into `outputs/accuracy/`

Expected outputs:
- `outputs/accuracy/qwen3-8b-non-reasoning_accuracy_judge_batched.json`
- `outputs/accuracy/qwen3-8b-reasoning_accuracy_judge_batched.json`
- `outputs/accuracy/llama-3-1-8b-instruct_accuracy_judge_batched.json`

## How To Interpret The Baselines

You should look at two axes:

1. Diversity
- from `*_baseline_deviation_metrics.json`
- key values:
  - `average_cosine_deviation_to_centroid`
  - `max_cosine_deviation_to_centroid`

2. Accuracy
- from `*_accuracy_judge_batched.json`
- key value:
  - average judge score across all scored responses

Interpretation:
- higher diversity is better only if accuracy does not collapse
- if the reasoning variant has higher deviation and similar accuracy, it is a stronger baseline
- if diversity rises but judge-based prompt adherence falls sharply, the model may just be drifting or becoming noisy

## Post-Training Status

Qwen3 DivPO-style post-training is now split into:
- [build_divpo_pairs.py](/Users/annikaks/Desktop/_stanford/cs224N-NLPwDL/project_LMDiversity/build_divpo_pairs.py)
- [modal_build_divpo_pairs.py](/Users/annikaks/Desktop/_stanford/cs224N-NLPwDL/project_LMDiversity/modal_build_divpo_pairs.py)
- [train_divpo_qwen_tinker.py](/Users/annikaks/Desktop/_stanford/cs224N-NLPwDL/project_LMDiversity/train_divpo_qwen_tinker.py)
- [judge_divpo_pairs.py](/Users/annikaks/Desktop/_stanford/cs224N-NLPwDL/project_LMDiversity/judge_divpo_pairs.py)
- [modal_judge_divpo_pairs.py](/Users/annikaks/Desktop/_stanford/cs224N-NLPwDL/project_LMDiversity/modal_judge_divpo_pairs.py)

The intended workflow is:
- uses prompts `0..26` for training
- uses the unmodified base prompt plus all `10` seed modifiers
- generates batched candidate responses for each prompt-condition on Modal
- builds preference pairs from a set-aware reward on Modal:
  - prompt-adherence proxy from prompt/response embedding cosine similarity
  - marginal diversity contribution relative to already-selected responses
- then trains a Qwen3-8B LoRA adapter on Tinker with a DPO loss

It supports:
- `non_reasoning`
- `reasoning`
- or both sequentially

Build pairwise data locally:

```bash
python build_divpo_pairs.py --thinking-mode both
```

Build pairwise data on Modal:

```bash
modal run -d modal_build_divpo_pairs.py --thinking-mode both
```

Then train one mode on Tinker:

```bash
python train_divpo_qwen_tinker.py --thinking-mode non_reasoning
python train_divpo_qwen_tinker.py --thinking-mode reasoning
```

If you want to annotate existing pairs with LLM-as-judge accuracy before pruning/training:

```bash
python judge_divpo_pairs.py \
  --input outputs/divpo/qwen3-8b-non-reasoning-divpo/preference_pairs.jsonl \
  --annotated-output outputs/divpo/qwen3-8b-non-reasoning-divpo/preference_pairs_with_llm_accuracy.jsonl \
  --judge-provider anthropic \
  --judge-model claude-opus-4-6
```

Or on Modal:

```bash
modal run -d modal_judge_divpo_pairs.py \
  --input-path /root/project/outputs/divpo/qwen3-8b-non-reasoning-divpo/preference_pairs.jsonl \
  --annotated-output-path /root/project/outputs/divpo/qwen3-8b-non-reasoning-divpo/preference_pairs_with_llm_accuracy.jsonl \
  --judge-provider anthropic \
  --judge-model claude-opus-4-6 \
  --rpm 5
```

Control checkpoint frequency with:

```bash
python train_divpo_qwen_tinker.py --thinking-mode reasoning --save-every-steps 25
```

Outputs are written under:
- `outputs/divpo/qwen3-8b-non-reasoning-divpo/`
- `outputs/divpo/qwen3-8b-reasoning-divpo/`

Key artifacts:
- `candidates.jsonl`
- `preference_pairs.jsonl`
- `preference_pairs_tinker_ready.json`
- `training_metrics.json`
- `tinker_training_metrics.json`
- `tinker_checkpoints.json`

If you build pairs on Modal, download the resulting pair files before Tinker training. The important file is:
- `outputs/divpo/qwen3-8b-<mode>-divpo/preference_pairs.jsonl`

## Planned Post-Training Workflow

Once DivPO training is added, the intended workflow should be:

1. Use prompts `1..26` to build candidate generations for training.
2. Construct synthetic preference pairs using:
   - quality / prompt adherence
   - diversity contribution in embedding space
3. Fine-tune one or more open models with a preference-optimization objective.
4. Re-run generation on prompts `28..33`.
5. Re-run:
   - `python pipeline.py analyze-embeddings`
   - `python pipeline.py analyze-baseline-deviation`
   - `python run_accuracy_batched.py ...`
6. Compare post-trained metrics against the baseline metrics.

For the currently implemented Qwen path, the intended workflow is:
1. Run `modal run -d modal_build_divpo_pairs.py --thinking-mode both`
2. Download `preference_pairs.jsonl` for each mode into `outputs/divpo/...`
3. Run `python train_divpo_qwen_tinker.py --thinking-mode non_reasoning`
4. Run `python train_divpo_qwen_tinker.py --thinking-mode reasoning`
5. Re-run generation on prompts `28..33` using the trained Tinker checkpoint
6. Re-run:
   - `python pipeline.py analyze-embeddings`
   - `python pipeline.py analyze-baseline-deviation`
   - `python run_accuracy_batched.py ...`
7. Compare post-trained metrics against the baseline metrics.

## Useful Commands

Run embedding/diversity analysis on Modal:

```bash
modal run -d modal_analyze_baselines.py --task diversity
```

For Modal runs, make sure the remote environment has the required API keys:
- embeddings default to `intfloat/e5-large` on Hugging Face, so `OPENAI_API_KEY` is no longer required unless you explicitly switch back to OpenAI embeddings
- Anthropic judge: `ANTHROPIC_API_KEY`
- Gemini judge: `GEMINI_API_KEY`
- xAI judge: `XAI_API_KEY`

Run accuracy judging on Modal:

```bash
modal run -d modal_analyze_baselines.py \
  --task accuracy \
  --judge-provider anthropic \
  --judge-model claude-opus-4-6 \
  --batch-size 4 \
  --rpm 5
```

Run both on Modal:

```bash
modal run -d modal_analyze_baselines.py \
  --task both \
  --judge-provider anthropic \
  --judge-model claude-opus-4-6 \
  --batch-size 4 \
  --rpm 5
```

If you only want to analyze Qwen files:

```bash
modal run -d modal_analyze_baselines.py \
  --task accuracy \
  --judge-provider anthropic \
  --judge-model claude-opus-4-6 \
  --generation-glob '*qwen3*.json'
```

Estimate accuracy-judge runtime without scoring:

```bash
python run_accuracy_batched.py \
  --judge-provider anthropic \
  --judge-model claude-opus-4-6 \
  --batch-size 4 \
  --generation-glob '*qwen3*.json' \
  --estimate-only
```

Test one judge call before a full run:

```bash
python test_accuracy_judge_batched.py \
  --judge-provider anthropic \
  --judge-model claude-opus-4-6 \
  --batch-size 4
```

## Files

- [pipeline.py](/Users/annikaks/Desktop/_stanford/cs224N-NLPwDL/project_LMDiversity/pipeline.py): embedding analysis, baseline deviation analysis, legacy API generation
- [generate_open_source.py](/Users/annikaks/Desktop/_stanford/cs224N-NLPwDL/project_LMDiversity/generate_open_source.py): local/HF generation for post-trainable models
- [modal_generate_open_source.py](/Users/annikaks/Desktop/_stanford/cs224N-NLPwDL/project_LMDiversity/modal_generate_open_source.py): Modal wrapper for the open-source generator with persistent volumes and resume support
- [run_accuracy_batched.py](/Users/annikaks/Desktop/_stanford/cs224N-NLPwDL/project_LMDiversity/run_accuracy_batched.py): prompt-adherence accuracy scoring
- [test_accuracy_judge_batched.py](/Users/annikaks/Desktop/_stanford/cs224N-NLPwDL/project_LMDiversity/test_accuracy_judge_batched.py): judge smoke test
- [creative_writing_prompts_v3.json](/Users/annikaks/Desktop/_stanford/cs224N-NLPwDL/project_LMDiversity/creative_writing_prompts_v3.json): source prompt dataset
