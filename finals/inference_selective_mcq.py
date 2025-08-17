import argparse
import json
import os
import time
from typing import Any, Dict, List
from tqdm import tqdm
from openai import OpenAI
import random
import wandb

def parse_arguments():
    parser = argparse.ArgumentParser(description='VLM Multi-GPU Inference using OpenAI API')
    parser.add_argument('--model', type=str, required=True, help='VLMs')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to input data JSON file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output JSON file')
    parser.add_argument('--max_model_len', type=int, default=12288,
                        help='Maximum model length')
    parser.add_argument('--num_images_per_prompt', type=int, default=6,
                        help='Maximum number of images per prompt')
    # hyperparameters
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.2,
                        help='Top-p for sampling')
    parser.add_argument('--max_tokens', type=int, default=512,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--api_base', type=str, default="http://localhost:8000/v1",
                        help='vLLM API base URL')
    # NEW: Add prompting strategy option
    parser.add_argument('--prompting_strategy', type=str, default="cod_adaptive", 
                        choices=["original", "cod_adaptive", "selective_prompting"],
                        help='Prompting strategy to use')
    return parser.parse_args()

# chain of Draft prompting functions
def get_question_category(question: str) -> str:
    """Simple question categorization based on keywords"""
    question_lower = question.lower()
    perception_keywords = ['following', 'select']
    if any(kw in question_lower for kw in perception_keywords):
        return 'perception_mcq'
    else:
        return 'other'

def get_system_prompt(strategy: str, question: str = "") -> str:
    """Get system prompt based on strategy"""
    base_context = """You are a helpful autonomous driving assistant that can answer questions about images and videos. You are provided 1600x900 resolution images from multi-view sensors ordered as [CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_RIGHT, CAM_BACK_LEFT]. The object coordinates are provided in the format of <id, camera_view, x, y>, where the coordinate (x,y) is the center of the bounding box of the object."""
    if strategy == "original":
        return base_context + """"""
    elif strategy == "cod_adaptive":
        return base_context + """

Use concise Chain of Draft reasoning:

For MCQs: Quick visual scan → Eliminate wrong options → Answer
For Perception VQAs: Locate objects → Key attributes → Brief description  
For Predictions: Current state → Physics/constraints → Likely outcome
For Planning: Safety check → Traffic rules → Best action

Keep reasoning minimal but sufficient. Only reference visible elements."""
    elif strategy == "selective_prompting":
        category = get_question_category(question)
        if category == 'perception_mcq':
            return """You are a helpful autonomous driving assistant that can answer questions about images and videos. You are provided 1600x900 resolution images from multi-view sensors ordered as [CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_RIGHT, CAM_BACK_LEFT]. The object coordinates are provided in the format of <id, camera_view, x, y>, where\nthe coordinate (x,y) is the center of the bounding box of the object.
##Step-by-Step Thinking:
Read and understand the user's question.
Scan each relevant image (if paths exist) to detect visual elements needed.
Cross-reference objects and their coordinates across views.
Infer actions, relationships, or descriptions based on visuals.
If needed, reason using multi-step logic or temporal alignment.

## Output Format:
Answer: <final answer>
Reasoning: <your chain-of-thought reasoning steps>
Image References: <list relevant camera views, and object ids/coordinates if used>
Confidence: <High / Medium / Low>

Be concise and avoid hallucinations. Only reference objects actually visible in the images provided
"""
        elif category == 'other':
            return """You are a helpful autonomous driving assistant that can answer questions about images and videos. You are provided 1600x900 resolution images from multi-view sensors ordered as [CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_RIGHT, CAM_BACK_LEFT]. The object coordinates are provided in the format of <id, camera_view, x, y>, where
the coordinate (x,y) is the center of the bounding box of the object.\n\nLet's first reason step-by-step:

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
    return get_system_prompt("original", question)

class VLMAPIInference:
    def __init__(self, model_name: str, api_base: str, temperature: float, 
                 top_p: float, max_tokens: int, prompting_strategy: str = "cod_adaptive"):
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=api_base
        )
        self.model = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.prompting_strategy = prompting_strategy

    def process_sample(self, question: str, img_paths: Dict[str, str]) -> str:
        system_prompt = get_system_prompt(self.prompting_strategy, question)
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        content = []
        for camera_view, img_path in img_paths.items():
            try:
                if any(img_path.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.webm']):
                    print(f"Warning: Video input detected: {img_path}. Video processing is disabled.")
                    continue
                if not os.path.exists(img_path):
                    print(f"Warning: Image file not found: {img_path}. Skipping.")
                    continue
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"file://{os.path.abspath(img_path)}"
                    }
                })
            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}. Skipping.")
                continue
        if not content:
            return "Error: No valid images found to process."
        content.append({
            "type": "text",
            "text": question
        })
        messages.append({
            "role": "user",
            "content": content
        })
        try:
            max_retries = 3
            retry_delay = 1
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise 
                    print(f"API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
        except Exception as e:
            error_msg = f"Error calling API: {str(e)}"
            print(error_msg)
            return error_msg
            
# the rest below is mosly unchanged from baseline


def load_or_create_output(output_path: str) -> List[Dict[str, Any]]:
    """Load existing output if it exists, or create a new output file."""
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                existing_data = json.load(f)
            print(f"Loaded {len(existing_data)} existing results from {output_path}")
            return existing_data
        except Exception as e:
            print(f"Error loading existing output file: {str(e)}. Starting fresh.")
            return []
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        return []

def save_output(output_path: str, data: List[Dict[str, Any]]):
    """Save output data to file."""
    temp_path = output_path + '.tmp'
    try:
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, output_path)
    except Exception as e:
        print(f"Error saving output: {str(e)}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

def process_qa_data(vlm: VLMAPIInference, data: List[Dict[str, Any]], output_path: str) -> List[Dict[str, Any]]:
    """Process QA data and generate answers, saving results in real-time."""
    # Load existing results or create new output file
    output_data = load_or_create_output(output_path)
    
    # Create set of already processed sample IDs
    processed_ids = {sample.get('id') for sample in output_data}
    
    # Filter out already processed samples
    remaining_data = [sample for sample in data if sample.get('id') not in processed_ids]
    
    if not remaining_data:
        print("All samples have already been processed!")
        return output_data
    
    # Process each remaining sample
    with tqdm(total=len(remaining_data), desc="Processing samples") as pbar:
        for sample in remaining_data:
            output_sample = sample.copy()
            
            answer = vlm.process_sample(sample['question'], sample['img_paths'])
            
            output_sample['answer'] = answer
            
            output_data.append(output_sample)
            
            save_output(output_path, output_data)
            
            pbar.update(1)
    
    return output_data

def main():
    args = parse_arguments()

    # Initialize VLM API client with prompting strategy
    vlm = VLMAPIInference(
        model_name=args.model,
        api_base=args.api_base,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        prompting_strategy=args.prompting_strategy # added
    )
    
    print(f"Using prompting strategy: {args.prompting_strategy}")
    
     # Initialize wandb
    run = wandb.init(
    project="robosense-track1",  # Customize project name
    entity="moely-university-of-virginia-seas",   # replace with your wandb username/teamr
    config={
        "model": args.model,
        "max_model_len": args.max_model_len,
        "num_images_per_prompt": args.num_images_per_prompt,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "input_data": args.data,
        })
    # Load input data
    print(f"Loading input data from {args.data}")
    with open(args.data, 'r') as f:
        data = json.load(f)

    # Process data and generate answers
    print("Processing data and generating answers...")
    output_data = process_qa_data(vlm, data, args.output)
  
    print("Done!")

if __name__ == '__main__':
    main()