#!/usr/bin/env python3
import argparse
import json
import os
import random
from datetime import datetime
import sys
os.chdir('//home/agauba/ICCV/AgMMU/scoring_eval_pipeline')
sys.path.append('/home/agauba/ICCV/AgMMU/scoring_eval_pipeline') 
import utils 

def format_options(q):
    options = q['options']
    st = ""
    for option, letter in zip(options, ["A.", "B.", "C.", "D."]):
        if option == q['answer']:
            q['letter'] = letter
        st += f"{letter} {option}\n"
    return st

def run_llms(prompt, img, q): ## MODIFY THIS TO RUN YOUR LLM
    system = "You are a helpful AI assistant."
    gpt = utils.exponential_backoff(utils.chat_gpt, system, prompt, img)
    q['answer'] = gpt
 

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

    for q in data:
        if q['faq-id'] in seen_ids:
            continue

        try:
            q['llm_answers'] = llm_map
            img_path = os.path.join(image_dir, f"{q['faq-id']}-1.jpg")

            for llm in q['llm_answers']:
                prefix = q['agmmu_question'].get('question_background', '')
                qtype = q['qtype']
                if 'mcq' in llm:
                    prompt = (
                        f"{prefix}{q['agmmu_question']['question']}\n"
                        f"Options:\n{format_options(q['agmmu_question'])}\n"
                        "Choose the letter corresponding with the correct answer. Only output the single letter."
                    )
                else:
                    if qtype in ['disease/issue identification', 'insect/pest', 'species']:
                        prompt = f"Question: {q['agmmu_question']['question']} Answer in 1-3 words."
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
            print("Error:", e)
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
        llm_map={"gpt-4o-oeq": {}, "gpt-4o-mcq":{}}, ## MODIFY THIS FOR YOUR MODEL
        output=args.output_path,
        image_dir=args.image_dir
    )

if __name__ == "__main__":
    main()
