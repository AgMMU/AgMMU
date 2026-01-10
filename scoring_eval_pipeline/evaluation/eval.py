#!/usr/bin/env python3
import argparse
import json
import os
import random
from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils 

def format_options(q):
    options = q['options']
    st = ""
    for option, letter in zip(options, ["A.", "B.", "C.", "D."]):
        if option == q['answer']:
            q['letter'] = letter
        st += f"{letter} {option}\n"
    return st

def run_llms(prompt, img, q, model_fn=None):
    system = "You are a helpful AI assistant."
    if model_fn is None:
        model_fn = utils.chat_gemini
    response = utils.exponential_backoff(model_fn, system, prompt, img)
    q['answer'] = response
 

def add_item_to_json(file_path, new_item):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as file:
            data = json.load(file)
    else:
        data = []

    if isinstance(new_item, list):
        data.extend(new_item)
    else:
        data.append(new_item)

    with open(file_path, 'w') as file:
        json.dump(data, file)

def eval_data(data, llm_map, output, image_dir):
    seen_ids = set()
    if os.path.exists(output):
        with open(output, 'r') as file:
            existing = json.load(file)
        seen_ids = {item['faq-id'] for item in existing}

    from tqdm import tqdm
    for q in tqdm(data, desc="Evaluating"):
        if q['faq-id'] in seen_ids:
            continue

        try:
            q['llm_answers'] = {k: {} for k in llm_map}
            faq_id = q['faq-id']
            img_path = os.path.join(image_dir, str(faq_id), f"{faq_id}_1.png")

            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue

            for llm in q['llm_answers']:
                prefix = q.get('question_background', '')
                qtype = q['qtype']
                if 'mcq' in llm:
                    prompt = (
                        f"{prefix}{q['question']}\n"
                        f"Options:\n{format_options(q)}\n"
                        "Choose the letter corresponding with the correct answer. Only output the single letter."
                    )
                else:
                    if qtype in ['disease/issue identification', 'insect/pest', 'species']:
                        prompt = f"Question: {q['question']} Answer in 1-3 words."
                    elif qtype == 'management instructions':
                        prompt = "What is the recommended management strategy for the issue seen in this image?\nBe descriptive."
                    elif qtype == 'symptom/visual description':
                        prompt = "What visual features can be seen in this image?\nBe descriptive."
                    else:
                        print("Unknown qtype:", qtype)
                        continue
                    prompt = prefix + prompt

                run_llms(prompt, img_path, q['llm_answers'][llm])

            add_item_to_json(output, q)

        except Exception as e:
            print(f"Error on {q.get('faq-id')}: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description="Evaluate visual symptom questions without RAG.")
    parser.add_argument("--data_path", required=True, help="Path to main data JSON.")
    parser.add_argument("--output_path", required=True, help="Path to save output JSON.")
    parser.add_argument("--image_dir", required=True, help="Directory with image files.")
    args = parser.parse_args()

    with open(args.data_path, 'r') as f:
        data = json.load(f)

    random.shuffle(data)

    eval_data(
        data=data,
        llm_map={"gemini-3-flash-oeq": {}, "gemini-3-flash-mcq": {}},
        output=args.output_path,
        image_dir=args.image_dir
    )

if __name__ == "__main__":
    main()
