import json
import os
import sys
from statistics import harmonic_mean
import argparse
import utils


with open('scoring_eval_pipeline/supporting_files/multi_statement.json') as file:
    multi = json.load(file)
with open('scoring_eval_pipeline/supporting_files/few_word_examples.json') as file:
    few_word_examples = json.load(file)


def score_pipeline(data, output):
    ids = []
    if os.path.exists(output):
        with open(output) as file:
            ids_file = json.load(file)
        ids = [i['faq-id'] for i in ids_file]

    for q in data:
        if q['faq-id'] in ids:
            continue
        try:
            for llm in q['llm_answers']:
                # if "gpt" in llm:
                #     continue
                # question_block = q['agmmu_question']
                question_block = q
                try:
                    if 'mcq' in llm:
                        q['llm_answers'][llm]['score'] = score_mcq({question_block['letter']: 1}, q['llm_answers'][llm]['answer'])
                    elif q['qtype'] in ['management instructions', 'symptom/visual description']:
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
                    print(f"error: {e}")
                    continue

            utils.add_item_to_json(output, q)
        except Exception as e:

            continue


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
    response = utils.exponential_backoff(utils.chat_gpt, system, prompt,None)

    filter_map = {"A": 1, "B": 0, "C": 0, "D": 0.5}
    return score_mcq(filter_map, response)
def create_multi_examples(qtype,question):
    st = ""
    for i in multi[qtype]:
        st += f"Gold Target:\n{i['expected']}\nPredicted Answer:\n{i['actual']}\nScoring:\n{i['score']}\n"
    return st


def score_multi_statement(qtype, actual,expected):
    
    if qtype == 'management instructions':
        question = "What is the recommended management strategy for the issue seen in this image?"
    else:
        question ="What visual features can be seen in this image?" 
    examples = create_multi_examples(qtype,question)
    system = f"""
    Your job is to grade student answers from the agriculture and biology domain. Your job is to look at a question, a gold target, and a predicted answer, and then assign grades to each statemetn in the response of  ['correct','partially correct', 'incorrect', 'missing', 'irrelevant'].
        - Correct is assigned to statements from the predicted answer that fully semantically map to a statement in the gold target.
        - Partially correct is assigned to statements which partially semantically map to a statement in the gold target.
        - Incorrect is assigned to statements from the predicted answer that directly semantically contradict a statement in the gold target.
        - Missing is assigned to statements in the gold target which haven't been mapped within correct,partially correct, or incorrect. 
        - Irrelevant is assigned to statements in the predicted answer which neither directly contradict nor corrospond in any way to statements in the gold target.

    EACH STATEMENT IN THE GOLD TARGET AND PREDICTED ANSWER SHOULD BE ASSIGNED TO EXACTLY ONE OF THESE CATEGORIES.
    Here are examples of correctly graded statements:
    {examples}

    Remember the following key points:
        - a statement is always partially correct if it has ANY overlap in content with the target


    Question: {question}
    Gold Target: {expected}
    Predicted Answer: {actual}

    Follow the format of the examples exactly. Output only a json with no additional text.
    """

    prompt = f"Question: {question}\nActual Statement:\n{actual}\n True Statement(s):\n{expected}\nScoring:\n"

    response = utils.exponential_backoff(utils.chat_gpt, system, prompt,None)

    response = utils.clean_response(response)
    

    return response


def score_mcq(target_map, predicted):
    cleaned = predicted.split(" ")[0].strip().lower().replace(".", "")
    for target in target_map:
        if cleaned == target.strip(" ")[0].lower().replace(".", "").strip():
            return {"accuracy": target_map[target]}
    return {"accuracy": 0}


def get_stats(data_path):
    with open(data) as f:
        data_path = json.load(data)
    scores = {} #to track scores by qtype
    overall_scores = {}  # To track overall metrics across all question types
    
    for i in data:
        for llm in i['llm_answers']:
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Path to input JSON")
    parser.add_argument("--output", "-o", required=True, help="Path to output JSON")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    score_pipeline(data, args.output)


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

    

