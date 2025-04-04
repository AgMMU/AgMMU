# AgMMU QA Evaluation and Scoring Overview

This is the scoring and evaluation script to evaluate and score your model on the AgMMU dataset. 


## AgMMU Evaluation
The `scoring_eval_pipeline/evaluation/eval.py` script evaluates and scores model outputs on the AgMMU dataset. It supports multiple-choice (MCQ) and open-ended question (OEQ) formats and expects outputs to follow a specific JSON structure.
## Setup

1. Download Images from Hugging Face (link)


## Usage
```bash
   python evaluate.py \
     --data_path /path/to/data.json \
     --output_path /path/to/output.json \
     --image_dir /path/to/image_directory/
```
### Arguments

- `--data_path`: Path to the input dataset JSON file.
- `--output_path`: Path to save the evaluation results.
- `--image_dir`: Directory containing input images named as `<faq-id>-1.jpg`.

## Modifying for Your LLM

Update the `run_llms()` function in `evaluate.py` to call your own model.

Edit the `llm_map` passed into `eval_data()` in `main()`:

   llm_map={"your-model-name-oeq": {}, "your-model-name-mcq": {}}

Note that model names must end with `-oeq` for open-ended or `-mcq` for multiple choice.






## AgMMU Scorer

The `scoring_eval_pipeline/scoring/score.py` script evaluates and scores model outputs on the AgMMU dataset.

##  How to Use

To run the scoring script, use the following command:

```bash
python score.py \
  --input_file input_file.json \
  --output_file output_file.json
```

## Input File Format

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

- The `answer` key inside each model entry should contain the model’s response.

## Output File

The `--output_file` will contain:
- Per-question scores
- A printed summary with harmonic means for overall evaluation metrics


