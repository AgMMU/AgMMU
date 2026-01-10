import json
import os
import re
import sys
from statistics import harmonic_mean
import argparse
from dotenv import load_dotenv
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

load_dotenv()

# Global config for Vertex AI (set from main)
_use_vertex = False
_output_lock = threading.Lock()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils


class StatementMapping(BaseModel):
    prediction: str
    gold_target: str


class MultiStatementScore(BaseModel):
    correct: list[StatementMapping]
    incorrect: list[StatementMapping]
    partially_correct: list[StatementMapping]
    missing: list[str]
    irrelevant: list[str]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, 'supporting_files/multi_statement.json')) as file:
    multi = json.load(file)
with open(os.path.join(SCRIPT_DIR, 'supporting_files/few_word_examples.json')) as file:
    few_word_examples = json.load(file)


def load_qa_information(source_path):
    """Load qa_information from full dataset, indexed by faq-id."""
    if not os.path.exists(source_path):
        print(f"Warning: {source_path} not found, qa_information will not be available")
        return {}
    with open(source_path) as f:
        data = json.load(f)
    return {item['faq-id']: item.get('qa_information') for item in data if 'qa_information' in item}


def score_single_question(q):
    """Score a single question. Returns the scored question or None on error."""
    try:
        for llm in q['llm_answers']:
            question_block = q
            try:
                if 'mcq' in llm:
                    q['llm_answers'][llm]['score'] = score_mcq({question_block['letter']: 1}, q['llm_answers'][llm]['answer'])
                elif q['qtype'] in ['management instructions', 'symptom/visual description']:
                    if 'qa_information' not in q:
                        print(f"Skipping {q['faq-id']}: missing qa_information for {q['qtype']}")
                        continue
                    qset = 'management instructions' if q['qtype'] == 'management instructions' else (
                        "image description" if 'image description' in q['qa_information'] else 'symptom description'
                    )
                    if not isinstance(q['qa_information'][qset], list):
                        print("not a list", qset, q['qa_information'][qset])
                    res = score_multi_statement(q['qtype'], q['llm_answers'][llm]['answer'], q['qa_information'][qset])
                    q['llm_answers'][llm]['score'] = res
                else:
                    q['llm_answers'][llm]['score'] = score_few_word(
                        question_block['question'],
                        question_block['answer'],
                        q['llm_answers'][llm]['answer'],
                        q['qtype']
                    )
            except Exception as e:
                print(f"error scoring {q['faq-id']}: {e}")
                continue
        return q
    except Exception as e:
        print(f"error processing {q.get('faq-id')}: {e}")
        return None


def score_pipeline(data, output, num_workers=8):
    ids = set()
    if os.path.exists(output):
        with open(output) as file:
            ids_file = json.load(file)
        ids = {i['faq-id'] for i in ids_file}

    # Filter to unscored questions
    to_score = [q for q in data if q['faq-id'] not in ids]

    if not to_score:
        print("All questions already scored.")
        return

    from tqdm import tqdm

    if num_workers == 1:
        # Sequential processing
        for q in tqdm(to_score, desc="Scoring"):
            result = score_single_question(q)
            if result:
                with _output_lock:
                    utils.add_item_to_json(output, result)
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(score_single_question, q): q for q in to_score}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring"):
                result = future.result()
                if result:
                    with _output_lock:
                        utils.add_item_to_json(output, result)


def score_few_word(question, target, predicted_answer, qtype):
    if predicted_answer.strip().lower().replace(".", "") == target.strip().lower().replace(".", ""):
        return {"accuracy": 1}

    examples = ""
    system = "You are a helpful AI assistant."
    for i, example in enumerate(few_word_examples[qtype]):
        examples += f"EXAMPLE {i + 1}:\n\nQuestion:\n{example['question']}\nGold Target:\n{example['target']}\nPredicted Answer:\n{example['actual']}\nGrade:\n{example['grade']}\n  -{example['rational']}\n"

    prompt = f"""
     Your job is to grade student answers from the agriculture and biology domain. Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ['CORRECT', 'INCORRECT', 'NOT ATTEMPTED', 'PARTIALLY CORRECT'].
     First, I will give examples of each grade, and then you will grade a new example.
     {examples}

    Remember the following key points:
        - a statement should be AT LEAST partially correct if the predicted answer is a subcategory of the gold target or the gold target is a subcategory of the predicted answer
        - a statement is always partially correct if it has ANY overlap in content with the target

    Grade the predicted answer of this new question as one of:
    A: CORRECT
    B: INCORRECT
    C: NOT_ATTEMPTED
    D: PARTIALLY CORRECT

    Question: {question}
    Gold Target: {target}
    Predicted Answer: {predicted_answer}

    Just return the letters "A", "B", "C", or "D", with no text around it.
    """
    response = utils.exponential_backoff(utils.chat_gemini, system, prompt, None, use_vertex=_use_vertex)

    filter_map = {"A": 1, "B": 0, "C": 0, "D": 0.5}
    return score_mcq(filter_map, response)
def create_multi_examples(qtype,question):
    st = ""
    for i in multi[qtype]:
        st += f"Gold Target:\n{i['expected']}\nPredicted Answer:\n{i['actual']}\nScoring:\n{i['score']}\n"
    return st


def score_multi_statement(qtype, actual, expected):

    if qtype == 'management instructions':
        question = "What is the recommended management strategy for the issue seen in this image?"
    else:
        question = "What visual features can be seen in this image?"
    examples = create_multi_examples(qtype, question)
    system = f"""
    Your job is to grade student answers from the agriculture and biology domain. Your job is to look at a question, a gold target, and a predicted answer, and then assign grades to each statement in the response of ['correct', 'partially_correct', 'incorrect', 'missing', 'irrelevant'].
        - correct is assigned to statements from the predicted answer that fully semantically map to a statement in the gold target.
        - partially_correct is assigned to statements which partially semantically map to a statement in the gold target.
        - incorrect is assigned to statements from the predicted answer that directly semantically contradict a statement in the gold target.
        - missing is assigned to statements in the gold target which haven't been mapped within correct, partially_correct, or incorrect.
        - irrelevant is assigned to statements in the predicted answer which neither directly contradict nor correspond in any way to statements in the gold target.

    EACH STATEMENT IN THE GOLD TARGET AND PREDICTED ANSWER SHOULD BE ASSIGNED TO EXACTLY ONE OF THESE CATEGORIES.
    Here are examples of correctly graded statements:
    {examples}

    Remember the following key points:
        - a statement is always partially_correct if it has ANY overlap in content with the target
    """

    prompt = f"Question: {question}\nGold Target:\n{expected}\nPredicted Answer:\n{actual}"

    response = utils.exponential_backoff(
        utils.chat_gemini, system, prompt, None,
        response_schema=MultiStatementScore, use_vertex=_use_vertex
    )

    # Parse JSON and convert from list format to expected dict format
    data = json.loads(response)
    result = {
        "correct": {m["prediction"]: m["gold_target"] for m in data.get("correct", [])},
        "incorrect": {m["prediction"]: m["gold_target"] for m in data.get("incorrect", [])},
        "partially correct": {m["prediction"]: m["gold_target"] for m in data.get("partially_correct", [])},
        "missing": data.get("missing", []),
        "irrelevant": data.get("irrelevant", []),
    }
    return result


def extract_mcq_answer(text):
    """Extract single letter answer from potentially verbose response."""
    text = text.strip()

    # Check if ends with single letter (often on own line)
    lines = text.split('\n')
    last_line = lines[-1].strip()
    if last_line.upper().replace('.', '') in ['A', 'B', 'C', 'D']:
        return last_line.upper().replace('.', '')

    # Already a single letter
    if text.upper().replace('.', '') in ['A', 'B', 'C', 'D']:
        return text.upper().replace('.', '')

    # Look for 'The answer is X' or 'correct answer is X'
    match = re.search(r'(?:the\s+)?(?:correct\s+)?answer(?:\s+is)?[:\s]+\**([A-D])\**', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Look for **X** pattern - take the LAST one as it's usually the conclusion
    matches = re.findall(r'\*\*([A-D])\*\*', text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    # First word/letter if starts with A-D
    first = text.split()[0] if text.split() else ""
    if first and first[0].upper() in ['A', 'B', 'C', 'D']:
        return first[0].upper()

    return None


def score_mcq(target_map, predicted):
    extracted = extract_mcq_answer(predicted)
    if extracted:
        for target in target_map:
            if extracted == target.strip()[0].upper().replace(".", ""):
                return {"accuracy": target_map[target]}
    return {"accuracy": 0}


def get_stats(data_path):
    with open(data_path) as f:
        data = json.load(f)
    scores = {} #to track scores by qtype
    overall_scores = {}  # To track overall metrics across all question types

    for i in data:
        for llm in i['llm_answers']:
            # Skip if no score (scoring failed for this question)
            if 'score' not in i['llm_answers'][llm]:
                continue

            # Initialize per-category metrics
            scores.setdefault(llm, {}).setdefault(i['qtype'], {"correct": 0, "total": 0, "partial": 0, "num_questions": 0})
            metrics = scores[llm][i['qtype']]

            # Initialize overall metrics for this LLM if not exists
            overall_scores.setdefault(llm, {"correct": 0, "total": 0, "partial": 0, "num_questions": 0})
            overall_metrics = overall_scores[llm]

            if 'mcq' in llm or i['qtype'] not in ['management instructions', 'symptom/visual description']:
                acc = i['llm_answers'][llm]['score'].get('accuracy', 0)
                if acc == 1:
                    metrics['correct'] += 1
                    overall_metrics['correct'] += 1
                elif acc == 0.5:
                    metrics['partial'] += 1
                    overall_metrics['partial'] += 1
                metrics['total'] += 1
                metrics['num_questions'] += 1
                overall_metrics['total'] += 1
                overall_metrics['num_questions'] += 1
            else:
                temp = i['llm_answers'][llm]['score']
                
                # For each gold target, track the best matching prediction
                gold_to_best_pred = {}
                
                # Process correct matches
                for pred, gold in temp['correct'].items():
                    if gold not in gold_to_best_pred:
                        gold_to_best_pred[gold] = (pred, 1)  # (prediction, score: 1 for correct)
                
                # Process partially correct matches (only if no correct match exists for this gold)
                for pred, gold in temp['partially correct'].items():
                    if gold not in gold_to_best_pred:
                        gold_to_best_pred[gold] = (pred, 0.5)  # (prediction, score: 0.5 for partial)
                
                # Count the best matches
                correct_count = sum(1 for _, score in gold_to_best_pred.values() if score == 1)
                partial_count = sum(1 for _, score in gold_to_best_pred.values() if score == 0.5)
                
                # Get all unique gold targets that should be matched
                all_gold_targets = set(temp['correct'].values()) | set(temp['partially correct'].values()) | set(temp['incorrect'].values()) | set(temp['missing'])
                num_statements = len(all_gold_targets) if all_gold_targets else 0
                
                if num_statements == 0:
                    continue
                
                # Update per-category metrics
                metrics['correct'] += correct_count / num_statements
                metrics['partial'] += partial_count / num_statements
                metrics['total'] += 1
                metrics['num_questions'] += 1
                
                # Update overall metrics
                overall_metrics['correct'] += correct_count / num_statements
                overall_metrics['partial'] += partial_count / num_statements
                overall_metrics['total'] += 1
                overall_metrics['num_questions'] += 1
    
    return scores, overall_scores

def calculate_harmonic_means(data):
    result = {}
    for model, categories in data.items():
        result[model] = {}
        for category, metrics in categories.items():
            correct = metrics['correct']
            total = metrics['num_questions']
            partial = metrics.get('partial', 0)
            metric1 = correct / total if total > 0 else 0
            metric2 = correct / (total - partial) if (total - partial) > 0 else 0
            result[model][category] = harmonic_mean([metric1, metric2]) if metric1 > 0 and metric2 > 0 else 0
    return result

def calculate_overall_accuracy(overall_scores):
    result = {}
    for model, metrics in overall_scores.items():
        correct = metrics['correct']
        total = metrics['num_questions']
        partial = metrics.get('partial', 0)
        metric1 = correct / total if total > 0 else 0
        metric2 = correct / (total - partial) if (total - partial) > 0 else 0
        result[model] = harmonic_mean([metric1, metric2]) if metric1 > 0 and metric2 > 0 else 0
    return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Path to input JSON")
    parser.add_argument("--output", "-o", required=True, help="Path to output JSON")
    parser.add_argument("--vertex", action="store_true", help="Use Vertex AI instead of API key")
    parser.add_argument("--workers", "-w", type=int, default=8, help="Number of parallel workers (default: 8)")
    parser.add_argument("--qa-source", default="data/6k_evalset_wbg.json",
                        help="Path to dataset with qa_information (default: data/6k_evalset_wbg.json)")
    args = parser.parse_args()

    # Set global Vertex AI config
    _use_vertex = args.vertex

    with open(args.input) as f:
        data = json.load(f)

    # Merge qa_information from source dataset if missing
    qa_lookup = load_qa_information(args.qa_source)
    merged_count = 0
    for item in data:
        if 'qa_information' not in item and item['faq-id'] in qa_lookup:
            item['qa_information'] = qa_lookup[item['faq-id']]
            merged_count += 1
    if merged_count > 0:
        print(f"Merged qa_information for {merged_count} items from {args.qa_source}")

    score_pipeline(data, args.output, num_workers=args.workers)

    stats, overall_stats = get_stats(args.output)
    harmonic_means = calculate_harmonic_means(stats)
    overall_accuracy = calculate_overall_accuracy(overall_stats)

    # Pretty-print results
    print("\n=== Overall Accuracy Scores ===")
    for model, score in overall_accuracy.items():
        print(f"{model}: {score:.4f}")

    print("\n=== Harmonic Mean Scores by Category ===")
    for model, categories in harmonic_means.items():
        print(f"\nModel: {model}")
        for category, score in categories.items():
            print(f"  {category}: {score:.4f}")

    

