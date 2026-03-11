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

Each selected prompt uses all available `seed_modifiers`.

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
- uses `3` prompt variants per prompt by default:
  - the unmodified base prompt
  - the first `2` seed-modified prompts
- generates `4` responses per prompt-condition by default
- batches generations locally with `--batch-size 4` by default
- writes one file per model to `outputs/generations/`

Explicit version:

```bash
python generate_open_source.py --preset tinker-baselines --max-modifiers 2 --num-samples 4 --batch-size 4
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
  --max-modifiers 2 \
  --num-samples 4 \
  --batch-size 4 \
  --max-tokens 700
```

Then Qwen reasoning:

```bash
modal run -d modal_generate_open_source.py \
  --model-spec 'qwen3-8b-reasoning|Qwen/Qwen3-8B|reasoning' \
  --max-modifiers 2 \
  --num-samples 4 \
  --batch-size 4 \
  --max-tokens 700
```

If you have gated-model access, Llama:

```bash
modal run -d modal_generate_open_source.py \
  --model-spec 'llama-3-1-8b-instruct|meta-llama/Llama-3.1-8B-Instruct|non_reasoning' \
  --max-modifiers 2 \
  --num-samples 4 \
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

Post-training / DivPO is **not implemented yet** in this repository.

That means there is currently **no command you can run yet** for:
- constructing DivPO preference pairs
- training a Qwen/Llama checkpoint with those pairs
- saving post-trained adapters
- re-running the same pipeline automatically on trained adapters

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

There is currently no runnable training command in this repo. The baseline pipeline is implemented; the post-training pipeline is not.

## Useful Commands

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
