#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from typing import Any, Dict, List
from collections import Counter

from tqdm import tqdm
from openai import OpenAI
import wandb

def parse_arguments():
    parser = argparse.ArgumentParser(description='VLM inference w/ selective prompting + MCQ self-consistency')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max_model_len', type=int, default=12288)
    parser.add_argument('--num_images_per_prompt', type=int, default=6)
    parser.add_argument('--api_base', type=str, default="http://localhost:8000/v1")
    parser.add_argument('--prompting_strategy', type=str, default="selective_prompting",
                        choices=["selective_prompting", "cod_adaptive", "original"])
    parser.add_argument('--mcq_self_consistency', type=int, default=1)
    #parser.add_argument('--mcq_sample_temperature', type=float, default=0.15)
    #parser.add_argument('--mcq_sample_top_p', type=float, default=0.2)
    #parser.add_argument('--mcq_sample_max_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top_p', type=float, default=0.2)
    parser.add_argument('--max_tokens', type=int, default=512)
    return parser.parse_args()

def get_question_category(question: str) -> str:
    question_lower = question.lower()
    perception_keywords = ['following', 'select']
    if any(kw in question_lower for kw in perception_keywords):
        return 'perception_mcq'
    else:
        return 'other'

def get_system_prompt(strategy: str, question: str = "") -> str:
    mcq_prompt = """You are a helpful autonomous driving assistant that can answer questions about images and videos. You are providing images from multi-view sensors ordered as [CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_RIGHT, CAM_BACK_LEFT]. The object coordinates are provided in the format of <id, camera_view, x, y>. The coordinate is the center of the bounding box where the image resolution is 1600x900.
"""
    other_prompt = """You are a helpful autonomous driving assistant that can answer questions about images and videos. You are provided 1600x900 resolution images from multi-view sensors ordered as [CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_RIGHT, CAM_BACK_LEFT]. The object coordinates are provided in the format of <id, camera_view, x, y>, where 
the coordinate (x,y) is the center of the bounding box of the object.

Let's first reason step-by-step:

1. Observe the scene and describe any potential obstacles or traffic agents.
2. List possible actions the ego vehicle might take (e.g., go straight, stop, turn).
3. Evaluate the safety and feasibility of each action.
4. Select the most reasonable action and estimate the probability it will be taken.

Draft answer based on the reasoning above:
[Insert draft response summarizing reasoning and giving action + probability.]

Now, refine the draft into a clear, concise answer with justification:
## Output Format:
answer + short explanation + probability estimate.

Be concise and avoid hallucinations"""
    if strategy == "selective_prompting":
        category = get_question_category(question)
        if category == 'perception_mcq':
            return mcq_prompt
        else:
            return other_prompt
    if strategy == "original":
        return other_prompt
    if strategy == "cod_adaptive":
        return other_prompt
    return other_prompt

def is_mcq(question: str) -> bool:
    q = question.lower()
    triggers = [
        "which of the following", "select the correct answer", "please select the correct answer",
        "following options", "options: a", "options: (a", " a) ", " b) ", " c) ", " d) "
    ]
    if any(t in q for t in triggers):
        return True
    if any(t in q for t in ['following', 'select']):
        return True
    return False

def is_planning(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ["should", "action", "plan", "yield", "merge", "turn", "stop", "probability"])

def is_perception_vqa(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ["describe", "object", "what is in", "what are the", "scene"])

def pick_decode_params(question: str):
    if is_mcq(question):
        return dict(temperature=0.20, top_p=0.20, max_tokens=512)
    if is_planning(question):
        return dict(temperature=0.20, top_p=0.40, max_tokens=768)
    if is_perception_vqa(question):
        return dict(temperature=0.15, top_p=0.30, max_tokens=512)
    return dict(temperature=0.20, top_p=0.20, max_tokens=512)

#LETTER_RE = re.compile(r'\b([A-D])\b', re.IGNORECASE)

#def extract_letter(text: str) -> str | None:
#    m = LETTER_RE.search(text or "")
#    return m.group(1).upper() if m else None

#def postprocess_answer(question: str, raw: str) -> str:
#    a = (raw or "").strip()
#    ans_lines = re.findall(r'(?i)^answer\s*:\s*([A-D])\b.*$', a, flags=re.MULTILINE)
#    if ans_lines and is_mcq(question):
#        return ans_lines[-1].upper()
#    if is_mcq(question):
#        letters = re.findall(r'\b([A-D])\b', a, flags=re.I)
#        if letters:
#            return letters[-1].upper()
#    a = re.split(r'(?:reasoning|image references|confidence)\s*:', a, flags=re.I)[0].strip()
#    a = re.sub(r'\s+', ' ', a)
#    return a[:300]
'''
def mcq_majority_letter(client: OpenAI, model: str, messages: List[Dict[str, Any]],
                        n: int, temp: float, top_p: float, max_toks: int) -> str | None:
    votes = []
    for _ in range(n):
        r = client.chat.completions.create(
            model=model, messages=messages,
            temperature=temp, top_p=top_p, max_tokens=max_toks
        )
        txt = (r.choices[0].message.content or "").strip()
        letter = extract_letter(txt)
        if letter:
            votes.append(letter)
    if not votes:
        return None
    return Counter(votes).most_common(1)[0][0]
'''
class VLMAPIInference:
    def __init__(self,
                 model_name: str,
                 api_base: str,
                 temperature: float,
                 top_p: float,
                 max_tokens: int,
                 prompting_strategy: str = "selective_prompting",
                 mcq_self_consistency: int = 1,
                 #mcq_sample_temperature: float = 0.7,
                 #mcq_sample_top_p: float = 0.9,
                 #mcq_sample_max_tokens: int = 32
                 ):
        self.client = OpenAI(api_key="EMPTY", base_url=api_base)
        self.model = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.prompting_strategy = prompting_strategy
        self.mcq_self_consistency = mcq_self_consistency
        #self.mcq_sample_temperature = mcq_sample_temperature
        #self.mcq_sample_top_p = mcq_sample_top_p
        #self.mcq_sample_max_tokens = mcq_sample_max_tokens

    def process_sample(self, question: str, img_paths: Dict[str, str]) -> str:
        messages = [{"role": "system", "content": get_system_prompt(self.prompting_strategy, question)}]
        content: List[Dict[str, Any]] = []
        for _camera, img_path in img_paths.items():
            try:
                if any(img_path.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.webm']):
                    continue
                if not os.path.exists(img_path):
                    continue
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"file://{os.path.abspath(img_path)}"}
                })
            except Exception:
                continue
        if not content:
            return "Error: No valid images found to process."
        content.append({"type": "text", "text": question})
        messages.append({"role": "user", "content": content})
        dec = pick_decode_params(question)
        #if is_mcq(question) and self.mcq_self_consistency and self.mcq_self_consistency > 1:
        #    letter = mcq_majority_letter(
        #        self.client, self.model, messages,
        #        n=self.mcq_self_consistency,
        #        temp=0.2,
        #        top_p=0.2,
        #        max_toks=48
        #    )
        #    if letter:
        #        return f"Answer: {letter}"
        max_retries, delay = 3, 1.0
        for attempt in range(max_retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=dec["temperature"],
                    top_p=dec["top_p"],
                    max_tokens=dec["max_tokens"]
                )
                return r.choices[0].message.content
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Error calling API: {e}"
                time.sleep(delay)
                delay *= 2.0

def load_or_create_output(output_path: str) -> List[Dict[str, Any]]:
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} existing results from {output_path}")
            return data
        except Exception as e:
            print(f"Error loading existing output: {e}. Starting fresh.")
            return []
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return []

def save_output(output_path: str, data: List[Dict[str, Any]]):
    tmp = output_path + ".tmp"
    try:
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, output_path)
    except Exception as e:
        print(f"Error saving output: {e}")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

def process_qa_data(vlm: VLMAPIInference, data: List[Dict[str, Any]], output_path: str) -> List[Dict[str, Any]]:
    out = load_or_create_output(output_path)
    processed_ids = {s.get('id') for s in out}
    remaining = [s for s in data if s.get('id') not in processed_ids]
    if not remaining:
        print("All samples already processed.")
        return out
    with tqdm(total=len(remaining), desc="Processing samples") as pbar:
        for sample in remaining:
            output_sample = sample.copy()
            raw = vlm.process_sample(sample['question'], sample['img_paths'])
            output_sample['answer'] = raw
            out.append(output_sample)
            save_output(output_path, out)
            pbar.update(1)
    return out

def main():
    args = parse_arguments()
    print(f"Using prompting strategy: {args.prompting_strategy}")
    vlm = VLMAPIInference(
        model_name=args.model,
        api_base=args.api_base,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        prompting_strategy=args.prompting_strategy,
        mcq_self_consistency=args.mcq_self_consistency,
        #mcq_sample_temperature=args.mcq_sample_temperature,
        #mcq_sample_top_p=args.mcq_sample_top_p,
        #mcq_sample_max_tokens=args.mcq_sample_max_tokens
    )
    print(f"Loading input data from {args.data}")
    with open(args.data, 'r') as f:
        data = json.load(f)
    print("Processing data and generating answers...")
    process_qa_data(vlm, data, args.output)
    print("Done!")
    run = wandb.init(
        project="robosense-track1",
        entity="moely-university-of-virginia-seas",
        config={
            "model": args.model,
            "max_model_len": args.max_model_len,
            "num_images_per_prompt": args.num_images_per_prompt,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "input_data": args.data,
            "prompting_strategy": args.prompting_strategy,
            "mcq_self_consistency": args.mcq_self_consistency,
        }
    )

if __name__ == '__main__':
    main()
