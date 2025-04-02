# AgMMU Evaluation Scorer

This script scores evaluation results for the **AgMMU** benchmark.

## Overview

The `score.py` script evaluates and scores model outputs on the AgMMU dataset. It supports multiple-choice (MCQ) and open-ended question (OEQ) formats and expects outputs to follow a specific JSON structure.

##  How to Use

To run the scoring script, use the following command:

```bash
python score.py \
  --input_file input_file.json \
  --output_file output_file.json
```

## 📥 Input File Format

The input file should be a JSON list of evaluated questions. Each item must include:

- `agmmu_question`: The original question object.
- `qtype`: Type of the question (e.g., `insect/pest`).
- `llm_answers`: A dictionary mapping LLM names to their answers.

### Example Format

```json
[
  {
    "agmmu_question": {
      "question": "What insect is indicated by the image?",
      "options": [
        "roseslug sawflies",
        "japanese beetle",
        "spotted lanternfly nymph",
        "gypsy moth larva"
      ],
      "answer": "roseslug sawflies",
      "question_background": "The following question has this background information:\nbackground info: plant is in a balcony container, has been sprayed with Neem oil with no effect, plant is Eden White Climber\nspecies: rose\nlocation: Montgomery County,Maryland\ntime: 2023-07-19 02:00:44\n",
      "letter": "A."
    },
    "qtype": "insect/pest",
    "llm_answers": {
      "gpt-4o-oeq": {
        "answer": "Aphids"
      },
      "gpt-4o-mcq": {
        "answer": "C"
      }
    }
  }
]
```

### Notes:
- `llm_answers` must include model names ending with `-oeq` for open-ended or `-mcq` for multiple choice.
- The `answer` key inside each model entry should contain the model’s response.

## Output File

The `--output_file` will contain:
- Per-question scores
- A printed summary with harmonic means for overall evaluation metrics


