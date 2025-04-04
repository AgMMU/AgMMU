import json
import os
import sys
from statistics import harmonic_mean
import argparse
os.chdir('//home/agauba/ICCV/AgMMU/scoring_eval_pipeline')
sys.path.append('/home/agauba/ICCV/AgMMU/scoring_eval_pipeline') 
import utils

handler = utils.ModelHandler()

with open('/home/agauba/ICCV/AgMMU/scoring_eval_pipeline/scoring/supporting_files/examples_s1.json') as file:
    examples_s1 = json.load(file)
with open('/home/agauba/ICCV/AgMMU/scoring_eval_pipeline/scoring/supporting_files/examples_s2.json') as file:
    examples_s2 = json.load(file)
with open('/home/agauba/ICCV/AgMMU/scoring_eval_pipeline/scoring/supporting_files/few_word_examples.json') as file:
    few_word_examples = json.load(file)


# Scoring functions
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
                if 'mcq' in llm:
                    q['llm_answers'][llm]['score'] = score_mcq({q['agmmu_question']['letter']: 1}, q['llm_answers'][llm]['answer'])
                elif q['qtype'] in ['management instructions', 'symptom/visual description']:
                    qset = 'management instructions' if q['qtype'] == 'management instructions' else (
                        "image description" if 'image description' in q['qa_information'] else 'symptom description'
                    )
                    if not isinstance(q['qa_information'][qset], list):
                        print("not a list", qset, q['qa_information'][qset])
                    q['llm_answers'][llm]['score'] = score_multi_statement(q['qtype'], q['llm_answers'][llm]['answer'], q['qa_information'][qset])
                else:
                    q['llm_answers'][llm]['score'] = score_few_word(
                        q['agmmu_question']['question'],
                        q['agmmu_question']['answer'],
                        q['llm_answers'][llm]['answer'],
                        q['qtype']
                    )

            utils.add_item_to_json(output, q)
        except Exception as e:
            print("BAD", e, q['faq-id'])
            continue


def score_few_word(question, target, predicted_answer, qtype):
    if question.strip().lower().replace(".", "") == target.strip().lower().replace(".", ""):
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
    response = handler.generate_response(system, prompt)
    filter_map = {"A": 1, "B": 0, "C": 0, "D": 0.5}
    return score_mcq(filter_map, response)


def score_multi_statement(qtype, predicted, expected_list):
    rational = {"correct": {}, "incorrect": {}, "partially correct": {}, "missing": [], "irrelevant": [], "repeat": {}}
    split = step_1(qtype, predicted)
    split = utils.clean_response(split)

    for statement in split:
        max_score = -1
        matched = ""
        repeat = {}
        for expected in expected_list:
            s2 = step_2(qtype, statement, expected)
            score = score_mcq({"A": 1, "B": 0, "C": -1, "D": 0.5}, s2)['accuracy']
            if score > max_score:
                max_score = score
                matched = expected
            elif score > 0:
                repeat[expected] = (statement, score)

        if max_score == -1:
            rational['irrelevant'].append(statement)
        elif max_score == 1:
            rational['correct'][statement] = matched
        elif max_score == 0.5:
            rational['partially correct'][statement] = matched
        elif max_score == 0:
            rational['incorrect'][statement] = matched

    ls = list(rational['correct'].values()) + list(rational['partially correct'].values()) + list(rational['incorrect'].values())
    for statement in expected_list:
        if statement not in ls:
            if statement in repeat:
                rational['repeat'][statement] = repeat[statement]
            else:
                rational['missing'].append(statement)

    return rational


def step_1(qtype, actual):
    examples = ""
    system = "You are a helpful AI assistant."
    for i, example in enumerate(examples_s1[qtype]):
        examples += f"EXAMPLE {i + 1}:\n\nORIGINAL: \n{example['actual']}\n\nSPLIT: \n{example['split']}\n\n"

    prompt = f"Your job is to split up a {qtype} response into multiple stand-alone {qtype} statements. Only include statements with informational content, not fluff or unrelated text.\nHere are some examples:\n{examples}\nORIGINAL:{actual}\nSPLIT:\n"
    return handler.generate_response(system, prompt)


def score_mcq(target_map, predicted):
    cleaned = predicted.split(" ")[0].strip().lower().replace(".", "")
    for target in target_map:
        if cleaned == target.strip(" ")[0].lower().replace(".", "").strip():
            return {"accuracy": target_map[target]}
    return {"accuracy": 0}


def step_2(qtype, actual, expected):
    system = "You are a helpful AI assistant."
    examples = ""
    for i in examples_s2[qtype]:
        examples += f"EXAMPLE:\nGOLD TARGET:\n{i['expected']}\nPREDICTED ANSWER:\n{i['actual']}\n{i['rational']}\n"

    prompt = f"""
     Your job is to grade student answers from the agriculture and biology domain. Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ['CORRECT', 'INCORRECT', 'IRRELEVANT', 'PARTIALLY CORRECT'].
     First, I will give examples of each grade, and then you will grade a new example.
     {examples}

    Remember:
        - Don't mark as incorrect unless it says something directly opposite to the target.
        - Always partially correct if there is any overlap.

    GOLD TARGET: {expected}
    PREDICTED ANSWER: {actual}

    Just return the letters "A", "B", "C", or "D", with no text around it.
    """
    return handler.generate_response(system, prompt)


# Evaluation
def move_duplicates_to_irrelevant(data):
    if 'irrelevant' not in data:
        data['irrelevant'] = []

    priorities = ["correct", "partially correct", "incorrect"]
    for i, higher in enumerate(priorities):
        higher_values = set(data[higher].values())
        for lower in priorities[i + 1:]:
            to_move = [k for k, v in data[lower].items() if v in higher_values]
            for k in to_move:
                data['irrelevant'].append(k)
                del data[lower][k]
    return data


def get_stats(data):
    scores = {}
    for i in data:
        for llm in i['llm_answers']:
            scores.setdefault(llm, {}).setdefault(i['qtype'], {"correct": 0, "total": 0, "partial": 0, "num_questions": 0})
            metrics = scores[llm][i['qtype']]

            if 'mcq' in llm or i['qtype'] not in ['management instructions', 'symptom/visual description']:
                acc = i['llm_answers'][llm]['score'].get('accuracy', 0)
                if acc == 1:
                    metrics['correct'] += 1
                elif acc == 0.5:
                    metrics['partial'] += 1
                metrics['total'] += 1
                metrics['num_questions'] += 1
            else:
                temp = i['llm_answers'][llm]['score']
                target_num = list(temp['correct'].values()) + list(temp['partially correct'].values()) + list(temp['incorrect'].values()) + temp['missing']
                if len(set(target_num)) < len(target_num):
                    move_duplicates_to_irrelevant(temp)
                num_statements = len(set(target_num))
                if num_statements == 0:
                    continue
                metrics['correct'] += len(temp['correct']) / num_statements
                metrics['partial'] += len(temp['partially correct']) / num_statements
                metrics['total'] += 1
                metrics['num_questions'] += 1
    return scores


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


def main():
    parser = argparse.ArgumentParser(description="Run AgMMU scoring pipeline.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to input evaluation JSON file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output scoring JSON file.')
    args = parser.parse_args()

    with open(args.input_file) as file:
        data = json.load(file)

    score_pipeline(data, args.output_file)


    with open(args.output_file) as file:
        archived_data = json.load(file)

    stats = get_stats(archived_data)
    harmonic_results = calculate_harmonic_means(stats)
    print("Harmonic Means:", harmonic_results)


if __name__ == "__main__":
    main()
