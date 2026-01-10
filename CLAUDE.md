# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgMMU (Agricultural Multimodal Understanding) is a benchmark for evaluating vision-language models (VLMs) in agriculture. The codebase provides tools for running inference and scoring model outputs on agricultural questions.

## Key Commands

### Install Dependencies
```bash
uv sync                       # Creates .venv and installs from pyproject.toml
cp .env.example .env          # Then add your GEMINI_API_KEY
```

### Download Dataset
```bash
bash download.sh              # Eval set only (~17.8 GB)
bash download.sh --full       # Include fine-tuning data (~550 GB)
```

### Run Inference
```bash
.venv/bin/python scoring_eval_pipeline/evaluation/eval.py \
  --data_path data/agmmu_eval.json \
  --output_path data/results.json \
  --image_dir data/copied_images
```

### Run Scoring
```bash
.venv/bin/python scoring_eval_pipeline/scoring/score.py \
  --input data/results.json \
  --output data/scored.json
```

## Architecture

### Pipeline Flow
1. **Evaluation** (`scoring_eval_pipeline/evaluation/eval.py`): Runs VLM inference on questions
2. **Scoring** (`scoring_eval_pipeline/scoring/score.py`): Scores model outputs against ground truth
3. **Utils** (`scoring_eval_pipeline/utils.py`): Shared utilities for API calls, image handling, model loading

### Question Types
- **MCQ**: Multiple-choice questions (model names must end with `-mcq`)
- **OEQ**: Open-ended questions (model names must end with `-oeq`)
  - Short answer types: `disease/issue identification`, `insect/pest`, `species`
  - Long answer types: `management instructions`, `symptom/visual description`

### Scoring Methods
- **MCQ**: Direct letter matching
- **Few-word OEQ**: LLM-as-judge grading (CORRECT, INCORRECT, NOT_ATTEMPTED, PARTIALLY_CORRECT)
- **Multi-statement OEQ**: Statement-level matching (correct, partially correct, incorrect, missing, irrelevant)

### Data Format
Model outputs must include:
- `agmmu_question`: Original question object with `question`, `options`, `answer`, `letter`
- `qtype`: Question type
- `llm_answers`: Dict mapping model names to `{"answer": "..."}` entries

### Adding Custom Models
1. Modify `run_llms()` in `eval.py` to call your model
2. Update `llm_map` in `main()` with your model name (must end with `-oeq` or `-mcq`)

## Important Files
- `scoring_eval_pipeline/scoring/supporting_files/few_word_examples.json`: Grading examples by question type
- `scoring_eval_pipeline/scoring/supporting_files/multi_statement.json`: Examples for multi-statement scoring
