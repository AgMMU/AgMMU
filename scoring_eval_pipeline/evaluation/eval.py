#!/usr/bin/env python3
import argparse
import json
import os
import random
from datetime import datetime
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

_output_lock = threading.Lock() 

def format_options(q):
    options = q['options']
    st = ""
    for option, letter in zip(options, ["A.", "B.", "C.", "D."]):
        if option == q['answer']:
            q['letter'] = letter
        st += f"{letter} {option}\n"
    return st

def run_llms(prompt, img, q, model_fn=None, model=None, use_vertex=False):
    system = "You are a helpful AI assistant."
    if model_fn is None:
        model_fn = utils.chat_gemini
    response = utils.exponential_backoff(model_fn, system, prompt, img, model=model, use_vertex=use_vertex)
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

def eval_single_question(q, llm_map, image_dir, model, use_vertex):
    """Evaluate a single question. Returns the evaluated question or None on error."""
    try:
        q['llm_answers'] = {k: {} for k in llm_map}
        faq_id = q['faq-id']
        img_path = os.path.join(image_dir, str(faq_id), f"{faq_id}_1.png")

        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            return None

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

            run_llms(prompt, img_path, q['llm_answers'][llm], model=model, use_vertex=use_vertex)

        return q
    except Exception as e:
        print(f"Error on {q.get('faq-id')}: {e}")
        return None


def eval_data(data, llm_map, output, image_dir, model=None, use_vertex=False, num_workers=8):
    seen_ids = set()
    if os.path.exists(output):
        with open(output, 'r') as file:
            existing = json.load(file)
        seen_ids = {item['faq-id'] for item in existing}

    # Filter to unevaluated questions
    to_eval = [q for q in data if q['faq-id'] not in seen_ids]

    if not to_eval:
        print("All questions already evaluated.")
        return

    from tqdm import tqdm

    if num_workers == 1:
        # Sequential processing
        for q in tqdm(to_eval, desc="Evaluating"):
            result = eval_single_question(q, llm_map, image_dir, model, use_vertex)
            if result:
                with _output_lock:
                    add_item_to_json(output, result)
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(eval_single_question, q, llm_map, image_dir, model, use_vertex): q
                for q in to_eval
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                result = future.result()
                if result:
                    with _output_lock:
                        add_item_to_json(output, result)

def main():
    parser = argparse.ArgumentParser(description="Evaluate visual symptom questions without RAG.")
    parser.add_argument("--data_path", required=True, help="Path to main data JSON.")
    parser.add_argument("--output_path", required=True, help="Path to save output JSON.")
    parser.add_argument("--image_dir", required=True, help="Directory with image files.")
    parser.add_argument("--model", default="gemini-3-pro-preview", help="Model to use for inference.")
    parser.add_argument("--vertex", action="store_true", help="Use Vertex AI instead of API key.")
    parser.add_argument("--workers", "-w", type=int, default=8, help="Number of parallel workers (default: 8)")
    args = parser.parse_args()

    with open(args.data_path, 'r') as f:
        data = json.load(f)

    random.shuffle(data)

    # Use model name for llm_map keys
    model_base = args.model.replace("-preview", "")
    llm_map = {f"{model_base}-oeq": {}, f"{model_base}-mcq": {}}

    eval_data(
        data=data,
        llm_map=llm_map,
        output=args.output_path,
        image_dir=args.image_dir,
        model=args.model,
        use_vertex=args.vertex,
        num_workers=args.workers
    )

if __name__ == "__main__":
    main()
