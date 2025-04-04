import json
import os
import requests
from openai import APIError, RateLimitError
from io import BytesIO
from PIL import Image
import re
from dotenv import load_dotenv
from openai import OpenAI
from bs4 import BeautifulSoup as bs
import time
import base64
import random
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm

def exponential_backoff(func, *args, max_retries=100, delay=1):
    cnt = 0
    while cnt < max_retries:
        try:
            return func(*args)
        except RateLimitError as e:
            print(f"Rate limit reached: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2
            cnt += 1
        except Exception as e:
            print(e)
            break
    raise Exception(f"Failed to execute {func.__name__} after {max_retries} retries.")

def load_image(image_file):

    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_file)
    return image


def add_item_to_json(file_path, new_item):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path) as file:
            data = json.load(file)
    else:
        with open(file_path, 'w') as json_file:
            json.dump([], json_file)
        data = []
    if isinstance(new_item,list):
        data.extend(new_item)
    else:
        data.append(new_item)
    
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def clean_response(response):
    response = re.sub(r"([a-zA-Z])'(s|t|m|r|ve|d|ll)", r"\1'\2", response)
    response = response.replace("'", '"')
    response = re.sub(r"^```json|```$", "", response, flags=re.MULTILINE)

    return json.loads(response)
import base64
from openai import OpenAI

def encode_image(image_path):
    """Encodes an image to Base64 for OpenAI API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def chat_gpt(system, prompt, image_path=None):
    client = OpenAI(
        api_key='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',
    )

    message = create_message(system, prompt, image_path)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=message,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def create_message(system, prompt, image_path):
    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ]
        }
    ]

    if system:
        message.insert(0, {
            "role": "system",
            "content": system
        })

    if image_path:
        base64_image = encode_image(image_path)
        message[1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    return message




class ModelHandler:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_response(self, system: str, prompt: str):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response
