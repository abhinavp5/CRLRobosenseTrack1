#!/usr/bin/env python3
"""

1. Temporal inference - for temporal sequence processing
2. Selective inference - for selective prompting
3. Selective MCQ inference - for MCQ-specific handling

Category Routing:
    - Perception-MCQ        → selective_mcq
    - Perception-VQA-Object → selective
    - Perception-VQA-Scene  → temporal
    - Prediction-MCQs       → temporal
    - Planning-VQA-Scene    → selective
    - Planning-VQA-Object   → selective

"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from openai import OpenAI
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'unified_inference_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CLASSIFICATION FUNCTIONS
# ============================================================================

def classify_question(question: str, category: str) -> str:
    """
    Classifies a question into a subcategory based on its content and main category.
    
    Args:
        question (str): The question text.
        category (str): The main category (e.g., "Perception", "Planning").

    Returns:
        str: The determined subcategory, or None if no match is found.
    """
    # Normalize category to handle variations
    category = category.lower()

    if category == "perception":
        if re.search(r"moving status of object.*options:", question, re.IGNORECASE):
            return "Perception-MCQ"
        elif re.search(r"visual description of <c[0-9]+,CAM_.*,.*,.*>", question, re.IGNORECASE):
            return "Perception-VQA-Object"
        elif re.search(r"important objects in the current scene", question, re.IGNORECASE):
            return "Perception-VQA-Scene"
            
    elif category == "prediction":
        # Only 1 prediction category
        return "Prediction-MCQs"

    elif category == "planning":
        if re.search(r"What actions could the ego vehicle take based on <c[0-9]+,CAM_.*,.*,.*>", question, re.IGNORECASE):
            return "Planning-VQA-Object"
        elif re.search(r"(safe actions|dangerous actions|comment on this scene)", question, re.IGNORECASE):
            return "Planning-VQA-Scene"
        else: 
            return "Planning-VQA-Object"
    
    # Return None if no subcategory matches
    return None

# ============================================================================
# SYSTEM PROMPTS AND HELPER FUNCTIONS
# ============================================================================

# System prompt for temporal VLM model
TEMPORAL_SYSTEM_PROMPT = """You are an advanced autonomous driving assistant analyzing temporal sequences of multi-view camera images. 

You are provided with a temporal sequence of {num_frames} frames, each containing images from 6 cameras:
- CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_RIGHT, CAM_BACK_LEFT

The frames are ordered from oldest to newest, with Frame {num_frames} being the current frame. Each frame is approximately 0.5 seconds apart.

When analyzing:
1. Consider temporal changes and motion patterns across frames
2. Track objects and their trajectories
3. Identify emerging situations and predict likely future states
4. Focus on safety-critical elements and their evolution over time

Object coordinates are in format <id, camera_view, x, y> where (x, y) is the bounding box center in 1600x900 resolution.
"""

def get_question_category(question: str) -> str:
    """Simple question categorization based on keywords for selective prompting"""
    question_lower = question.lower()
    perception_keywords = ['following', 'select']
    if any(kw in question_lower for kw in perception_keywords):
        return 'perception_mcq'
    else:
        return 'other'

def get_system_prompt(strategy: str, question: str = "") -> str:
    """Get system prompt based on strategy"""
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
    return other_prompt

def is_mcq(question: str) -> bool:
    """Check if question is multiple choice"""
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
    """Check if question is about planning"""
    q = question.lower()
    return any(k in q for k in ["should", "action", "plan", "yield", "merge", "turn", "stop", "probability"])

def is_perception_vqa(question: str) -> bool:
    """Check if question is perception VQA"""
    q = question.lower()
    return any(k in q for k in ["describe", "object", "what is in", "what are the", "scene"])

def pick_decode_params(question: str):
    """Pick decoding parameters based on question type"""
    if is_mcq(question):
        return dict(temperature=0.20, top_p=0.20, max_tokens=512)
    if is_planning(question):
        return dict(temperature=0.20, top_p=0.40, max_tokens=768)
    if is_perception_vqa(question):
        return dict(temperature=0.15, top_p=0.30, max_tokens=512)
    return dict(temperature=0.20, top_p=0.20, max_tokens=512)

# ============================================================================
# TEMPORAL INFERENCE CLASS
# ============================================================================

class TemporalVLMInference:
    """
    Main inference engine for temporal Vision-Language Model processing.
    
    This class handles the core functionality of processing temporal sequences
    of multi-camera images through a VLM model.
    """
    
    def __init__(self, model_name: str, api_base: str, temperature: float, 
                 top_p: float, max_tokens: int, num_temporal_frames: int,
                 temporal_strategy: str):
        """Initialize the temporal VLM inference engine."""
        logger.info(f"Initializing TemporalVLMInference with model: {model_name}")
        
        # Initialize OpenAI client for vLLM server communication
        self.client = OpenAI(
            api_key="EMPTY",  # vLLM doesn't require an API key
            base_url=api_base
        )
        
        # Store configuration parameters
        self.model = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.num_temporal_frames = num_temporal_frames
        self.temporal_strategy = temporal_strategy
        
        # Validate temporal strategy
        if temporal_strategy not in ['sequential', 'interleaved']:
            raise ValueError(f"Invalid temporal_strategy: {temporal_strategy}")
        
        logger.info("TemporalVLMInference initialized successfully")
        
    def extract_temporal_frames(self, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract temporal sequence of frames from sample data."""
        logger.debug(f"Extracting temporal frames for sample")
        temporal_frames = []
        
        # Get history frames (oldest to newest)
        history_frames = sample.get('history_frames', {})
        history_tokens = list(history_frames.keys())
        logger.debug(f"Found {len(history_tokens)} history frames available")
        
        # Take the most recent N-1 history frames (leaving room for current frame)
        num_history = min(self.num_temporal_frames - 1, len(history_tokens))
        logger.debug(f"Using {num_history} history frames + 1 current frame")
        
        if num_history > 0:
            # Get the last N-1 history frames (most recent)
            selected_history = history_tokens[-num_history:]
            logger.debug(f"Selected history tokens: {selected_history}")
            
            for token in selected_history:
                temporal_frames.append(history_frames[token])
                logger.debug(f"Added history frame: {token}")
        
        # Add current frame as the last frame (most recent)
        temporal_frames.append(sample['img_paths'])
        logger.debug(f"Added current frame as frame {len(temporal_frames)}")
        
        logger.info(f"Extracted {len(temporal_frames)} temporal frames successfully")
        return temporal_frames
    
    def format_temporal_prompt(self, frames: List[Dict[str, str]], question: str) -> List[Dict]:
        """Format temporal frames into a structured prompt for the VLM."""
        logger.info(f"Formatting temporal prompt with strategy: {self.temporal_strategy}")
        content = []
        
        if self.temporal_strategy == 'sequential':
            # Sequential strategy: Group by temporal frame
            logger.debug("Using sequential temporal strategy")
            
            for frame_idx, frame_imgs in enumerate(frames, 1):
                # Add frame header
                content.append({
                    "type": "text",
                    "text": f"\n--- Frame {frame_idx} of {len(frames)} ---"
                })
                logger.debug(f"Processing frame {frame_idx}/{len(frames)}")
                
                # Add all camera views for this temporal frame
                cameras_added = 0
                for camera_view in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                                   'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']:
                    if camera_view in frame_imgs:
                        img_path = frame_imgs[camera_view]
                        if os.path.exists(img_path):
                            # Add camera label
                            content.append({
                                "type": "text",
                                "text": f"[{camera_view}]"
                            })
                            # Add image reference
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": f"file://{os.path.abspath(img_path)}"}
                            })
                            cameras_added += 1
                        else:
                            logger.warning(f"Image not found: {img_path}")
                
                logger.debug(f"Added {cameras_added} camera views for frame {frame_idx}")
                        
        elif self.temporal_strategy == 'interleaved':
            # Interleaved strategy: Group by camera view
            logger.debug("Using interleaved temporal strategy")
            
            for camera_view in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                               'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']:
                # Add camera header
                content.append({
                    "type": "text",
                    "text": f"\n--- {camera_view} Temporal Sequence ---"
                })
                logger.debug(f"Processing camera view: {camera_view}")
                
                # Add temporal sequence for this camera
                frames_added = 0
                for frame_idx, frame_imgs in enumerate(frames, 1):
                    if camera_view in frame_imgs:
                        img_path = frame_imgs[camera_view]
                        if os.path.exists(img_path):
                            # Add temporal index label
                            content.append({
                                "type": "text",
                                "text": f"[T{frame_idx}]"
                            })
                            # Add image reference
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": f"file://{os.path.abspath(img_path)}"}
                            })
                            frames_added += 1
                        else:
                            logger.warning(f"Image not found: {img_path}")
                
                logger.debug(f"Added {frames_added} temporal frames for {camera_view}")
        
        # Add the question at the end
        content.append({
            "type": "text",
            "text": f"\n\nQuestion: {question}\n\nPlease analyze the temporal sequence and provide your answer:"
        })
        
        logger.info(f"Formatted prompt with {len(content)} content items")
        return content

    def process_sample(self, sample: Dict[str, Any]) -> str:
        """Process a single sample through the temporal VLM pipeline."""
        sample_id = sample.get('frame_token', sample.get('id', 'unknown'))
        logger.info(f"Processing sample: {sample_id}")
        start_time = time.time()
        
        try:
            # Extract temporal frames from the sample
            logger.debug("Extracting temporal frames")
            temporal_frames = self.extract_temporal_frames(sample)
            
            # Prepare messages with system prompt
            logger.debug("Preparing messages with system prompt")
            messages = [
                {"role": "system", "content": TEMPORAL_SYSTEM_PROMPT.format(
                    num_frames=len(temporal_frames)
                )}
            ]
            
            # Format temporal content according to strategy
            logger.debug("Formatting temporal content")
            content = self.format_temporal_prompt(temporal_frames, sample['question'])
            
            # Add user message with formatted content
            messages.append({
                "role": "user",
                "content": content
            })
            logger.debug(f"Prepared message with {len(content)} content items")
            
            # Call API with retry logic for robustness
            max_retries = 3
            retry_delay = 2  # Initial delay in seconds
            logger.info(f"Calling VLM API for sample {sample_id}")
            
            for attempt in range(max_retries):
                try:
                    logger.debug(f"API call attempt {attempt + 1}/{max_retries}")
                    api_start = time.time()
                    # Make API call to vLLM server
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens
                    )
                    
                    api_time = time.time() - api_start
                    logger.info(f"API call successful for {sample_id} (took {api_time:.2f}s)")
                    
                    # Extract and return the generated response
                    result = response.choices[0].message.content
                    total_time = time.time() - start_time
                    logger.info(f"Sample {sample_id} processed successfully in {total_time:.2f}s")
                    
                    return result
                    
                except Exception as e:
                    logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    
                    if attempt == max_retries - 1:
                        # Final attempt failed, propagate the error
                        logger.error(f"All retry attempts failed for sample {sample_id}")
                        raise
                    
                    # Exponential backoff for retry delay
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Double the delay for next attempt
                    
        except Exception as e:
            # Log the error and return an error message
            error_msg = f"Error processing sample {sample_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

# ============================================================================
# SELECTIVE INFERENCE CLASS
# ============================================================================

class SelectiveInference:
    """VLM API Inference with selective prompting strategy"""
    
    def __init__(self, model_name: str, api_base: str, temperature: float, 
                 top_p: float, max_tokens: int, prompting_strategy: str = "selective_prompting",
                 mcq_self_consistency: int = 1):
        self.client = OpenAI(api_key="EMPTY", base_url=api_base)
        self.model = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.prompting_strategy = prompting_strategy
        self.mcq_self_consistency = mcq_self_consistency

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

# ============================================================================
# SELECTIVE MCQ INFERENCE CLASS
# ============================================================================

class SelectiveMCQInference:
    """VLM API Inference optimized for MCQ questions"""
    
    def __init__(self, model_name: str, api_base: str, temperature: float, 
                 top_p: float, max_tokens: int, prompting_strategy: str = "selective_prompting"):
        self.client = OpenAI(api_key="EMPTY", base_url=api_base)
        self.model = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.prompting_strategy = prompting_strategy

    def process_sample(self, question: str, img_paths: Dict[str, str]) -> str:
        system_prompt = get_system_prompt(self.prompting_strategy, question)
        messages = [{"role": "system", "content": system_prompt}]
        content = []
        
        for camera_view, img_path in img_paths.items():
            try:
                if any(img_path.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.webm']):
                    logger.debug(f"Warning: Video input detected: {img_path}. Video processing is disabled.")
                    continue
                if not os.path.exists(img_path):
                    logger.debug(f"Warning: Image file not found: {img_path}. Skipping.")
                    continue
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"file://{os.path.abspath(img_path)}"}
                })
            except Exception as e:
                logger.debug(f"Error processing image {img_path}: {str(e)}. Skipping.")
                continue
                
        if not content:
            return "Error: No valid images found to process."
            
        content.append({"type": "text", "text": question})
        messages.append({"role": "user", "content": content})
        
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
                    logger.debug(f"API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
        except Exception as e:
            error_msg = f"Error calling API: {str(e)}"
            logger.error(error_msg)
            return error_msg

# ============================================================================
# UNIFIED DISPATCHER CLASS
# ============================================================================

class UnifiedDispatcher:
    """
    Unified dispatcher that routes questions to appropriate inference techniques.
    
    This class manages three different inference engines and routes each question
    to the most appropriate one based on category classification.
    """
    
    def __init__(self, args):
        """Initialize the unified dispatcher with all three inference engines."""
        logger.info("Initializing UnifiedDispatcher with three inference engines")
        
        # Initialize temporal inference engine
        logger.debug("Initializing temporal inference engine")
        self.temporal_engine = TemporalVLMInference(
            model_name=args.model,
            api_base=args.api_base,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            num_temporal_frames=args.num_temporal_frames,
            temporal_strategy=args.temporal_strategy
        )
        
        # Initialize selective inference engine with required parameters
        logger.debug("Initializing selective inference engine")
        self.selective_engine = SelectiveInference(
            model_name=args.model,
            api_base=args.api_base,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=512,  # Selective uses 512 max tokens
            prompting_strategy='selective_prompting',  # Always use selective_prompting
            mcq_self_consistency=1  # Always use 1
        )
        
        # Initialize selective MCQ inference engine
        logger.debug("Initializing selective MCQ inference engine")
        self.selective_mcq_engine = SelectiveMCQInference(
            model_name=args.model,
            api_base=args.api_base,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=512,  # MCQ uses 512 max tokens
            prompting_strategy='selective_prompting'  # Always use selective_prompting
        )
        
        logger.info("All three inference engines initialized successfully")
        
    def process_sample(self, sample: Dict[str, Any]) -> str:
        """Process a single sample by routing it to the appropriate inference engine."""
        sample_id = sample.get('frame_token', sample.get('id', 'unknown'))
        logger.info(f"Processing sample: {sample_id}")
        
        # Classify the question to determine subcategory
        category = sample.get('category', '')
        question = sample.get('question', '')
        
        subcategory = classify_question(question, category)
        logger.info(f"Sample {sample_id} classified as: {subcategory}")
        
        # Route to appropriate inference engine based on subcategory
        try:
            if subcategory == "Perception-MCQ":
                logger.debug(f"Routing {sample_id} to selective MCQ engine")
                # Use selective MCQ for Perception MCQ questions
                return self.selective_mcq_engine.process_sample(
                    question=sample['question'],
                    img_paths=sample['img_paths']
                )
                
            elif subcategory in ["Perception-VQA-Object", "Planning-VQA-Scene", "Planning-VQA-Object"]:
                logger.debug(f"Routing {sample_id} to selective engine")
                # Use selective inference for these categories
                return self.selective_engine.process_sample(
                    question=sample['question'],
                    img_paths=sample['img_paths']
                )
                
            elif subcategory in ["Perception-VQA-Scene", "Prediction-MCQs"]:
                logger.debug(f"Routing {sample_id} to temporal engine")
                # Use temporal inference for these categories
                return self.temporal_engine.process_sample(sample)
                
            else:
                # Default to selective inference for unmatched categories
                logger.warning(f"Unknown subcategory {subcategory} for {sample_id}, defaulting to selective")
                return self.selective_engine.process_sample(
                    question=sample['question'],
                    img_paths=sample['img_paths']
                )
                
        except Exception as e:
            error_msg = f"Error processing sample {sample_id} with {subcategory}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """Parse command-line arguments for the unified VLM inference dispatcher."""
    logger.debug("Parsing command-line arguments")
    parser = argparse.ArgumentParser(description='Unified Self-Contained VLM Inference')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-VL-72B-Instruct',
                        help='Model name (Qwen2.5-VL-32B or Qwen2.5-VL-72B)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to input data JSON file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output JSON file')
    
    # Temporal settings (for temporal inference)
    parser.add_argument('--num_temporal_frames', type=int, default=5,
                        help='Number of temporal frames to use (for temporal inference)')
    parser.add_argument('--temporal_strategy', type=str, default='sequential',
                        choices=['sequential', 'interleaved'],
                        help='How to arrange temporal frames in prompt')
    
    # GPU and model settings
    parser.add_argument('--tensor_parallel_size', type=int, default=4,
                        help='Number of GPUs for tensor parallelism')
    parser.add_argument('--max_model_len', type=int, default=32768,
                        help='Maximum model length')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                        help='GPU memory utilization for vLLM')
    
    # Hyperparameters (default values for all techniques)
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.2,
                        help='Top-p for sampling')
    parser.add_argument('--max_tokens', type=int, default=1024,
                        help='Maximum number of tokens to generate')
    
    # API settings
    parser.add_argument('--api_base', type=str, default="http://localhost:8000/v1",
                        help='vLLM API base URL')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for processing')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save results every N samples')
    
    # Logging settings
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging verbosity level')
    
    args = parser.parse_args()
    
    # Update logging level based on argument
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    logger.info(f"Logging level set to: {args.log_level}")
    
    logger.info("Command-line arguments parsed successfully")
    logger.debug(f"Arguments: {vars(args)}")
    
    return args

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_vllm_server(args):
    """Setup and verify vLLM server configuration for multi-GPU inference."""
    import subprocess
    
    logger.info("Checking vLLM server status...")
    
    # Check if vLLM server is already running
    try:
        logger.debug(f"Attempting to connect to vLLM server at {args.api_base}")
        test_client = OpenAI(api_key="EMPTY", base_url=args.api_base)
        test_client.models.list()
        logger.info("✓ vLLM server is already running and accessible")
        return
    except Exception as e:
        logger.info("vLLM server not detected, preparing launch command")
        logger.debug(f"Connection error: {str(e)}")
    
    # Prepare vLLM launch command
    vllm_cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--port", "8000",
        "--trust-remote-code",
        "--max-num-seqs", "32",
        "--enforce-eager",
    ]
    
    # Add Qwen2.5-VL specific settings
    if "Qwen2.5-VL" in args.model:
        logger.debug("Adding Qwen2.5-VL specific vision parameters")
        vllm_cmd.extend([
            "--image-input-type", "pixel_values",
            "--image-token-id", "151655",
            "--image-input-shape", "1,3,448,448",
            "--image-feature-size", "1176",
        ])
    
    # Display launch instructions
    logger.info("vLLM server launch command prepared")
    print("\n" + "="*70)
    print("vLLM Server Setup Required")
    print("="*70)
    print(f"\nLaunch command:\n{' '.join(vllm_cmd)}")
    print("\n" + "-"*70)
    print("Instructions:")
    print("1. Open a new terminal window")
    print("2. Run the command above to start the vLLM server")
    print("3. Wait for the server to fully initialize")
    print("4. Run this script again to process your data")
    print("="*70 + "\n")
    
    logger.info("Exiting - vLLM server needs to be started first")
    return

def save_checkpoint(output_path: str, data: List[Dict[str, Any]]):
    """Save data with atomic write operation for safety."""
    logger.debug(f"Saving checkpoint to {output_path}")
    
    # Write to temporary file first
    temp_path = output_path + '.tmp'
    try:
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Atomic rename to final location
        os.replace(temp_path, output_path)
        logger.debug(f"Checkpoint saved successfully ({len(data)} samples)")
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

def process_data_with_checkpointing(dispatcher: UnifiedDispatcher, data: List[Dict[str, Any]], 
                                   output_path: str, save_every: int = 10) -> List[Dict[str, Any]]:
    """Process data with periodic checkpointing for fault tolerance."""
    logger.info(f"Starting data processing with checkpointing (save_every={save_every})")
    
    # Load existing results if any
    if os.path.exists(output_path):
        logger.info(f"Found existing output file: {output_path}")
        with open(output_path, 'r') as f:
            output_data = json.load(f)
        
        processed_ids = {sample.get('frame_token', sample.get('id', i)) 
                        for i, sample in enumerate(output_data)}
        logger.info(f"Loaded {len(output_data)} previously processed samples")
    else:
        logger.info("Starting fresh processing (no existing output found)")
        output_data = []
        processed_ids = set()
    
    # Filter out already processed samples
    remaining_data = []
    for i, sample in enumerate(data):
        sample_id = sample.get('frame_token', sample.get('id', i))
        if sample_id not in processed_ids:
            if 'id' not in sample:
                sample['id'] = sample_id
            remaining_data.append(sample)
    
    logger.info(f"Found {len(remaining_data)} unprocessed samples out of {len(data)} total")
    
    if not remaining_data:
        logger.info("✓ All samples have already been processed!")
        print("\nAll samples already processed!")
        return output_data
    
    print(f"\nProcessing {len(remaining_data)} remaining samples...")
    logger.info(f"Beginning processing of {len(remaining_data)} samples")
    
    # Process samples with progress tracking
    start_time = time.time()
    with tqdm(total=len(remaining_data), desc="Processing samples") as pbar:
        for idx, sample in enumerate(remaining_data):
            sample_id = sample.get('id', f"sample_{idx}")
            logger.debug(f"Processing sample {idx+1}/{len(remaining_data)}: {sample_id}")
            
            try:
                # Process sample through unified dispatcher
                answer = dispatcher.process_sample(sample)
            except Exception as e:
                logger.error(f"Failed to process sample {sample_id}: {str(e)}")
                answer = f"Error: {str(e)}"
            
            # Add result to output data
            output_sample = sample.copy()
            output_sample['answer'] = answer
            output_sample['processed_at'] = datetime.now().isoformat()
            output_data.append(output_sample)
            
            logger.debug(f"Sample {sample_id} processed successfully")
            
            # Save checkpoint periodically
            if (idx + 1) % save_every == 0:
                save_checkpoint(output_path, output_data)
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                logger.info(f"Checkpoint saved at sample {idx + 1} ({rate:.2f} samples/sec)")
                print(f"\nCheckpoint saved at sample {idx + 1} ({rate:.2f} samples/sec)")
            
            pbar.update(1)
    
    # Final save
    save_checkpoint(output_path, output_data)
    
    # Log statistics
    total_time = time.time() - start_time
    avg_time = total_time / len(remaining_data) if remaining_data else 0
    logger.info(f"Processing complete: {len(remaining_data)} samples in {total_time:.2f}s")
    logger.info(f"Average processing time: {avg_time:.2f}s per sample")
    
    return output_data

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for the unified VLM inference dispatcher."""
    # Parse command-line arguments
    args = parse_arguments()
    
    logger.info("="*60)
    logger.info("Unified Self-Contained VLM Inference System Starting")
    logger.info("="*60)
    
    # Display configuration summary
    print("\n" + "=" * 60)
    print("Unified VLM Inference Configuration")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size} GPUs")
    print(f"Max Model Length: {args.max_model_len}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Max Tokens: {args.max_tokens}")
    print("\nRouting Configuration:")
    print("  Perception-MCQ        → Selective MCQ")
    print("  Perception-VQA-Object → Selective")
    print("  Perception-VQA-Scene  → Temporal")
    print("  Prediction-MCQs       → Temporal")
    print("  Planning-VQA-Scene    → Selective")
    print("  Planning-VQA-Object   → Selective")
    print("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"CUDA available with {num_gpus} GPU(s)")
        print(f"\nDetected {num_gpus} GPUs:")
        
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            logger.debug(f"GPU {i}: {gpu_name}, Memory: {gpu_memory:.1f} GB")
        
        if args.tensor_parallel_size > num_gpus:
            logger.warning(f"Tensor parallel size ({args.tensor_parallel_size}) exceeds available GPUs ({num_gpus})")
            print(f"\nWarning: Requested {args.tensor_parallel_size} GPUs but only {num_gpus} available!")
    else:
        logger.error("No CUDA GPUs detected")
        print("\nError: No GPUs detected! This system requires CUDA-capable GPUs.")
        sys.exit(1)
    
    # Setup or verify vLLM server
    logger.info("Setting up vLLM server")
    setup_vllm_server(args)
    
    # Initialize unified dispatcher
    logger.info("Initializing unified dispatcher")
    try:
        dispatcher = UnifiedDispatcher(args)
        logger.info("Unified dispatcher initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize dispatcher: {str(e)}")
        print(f"\nError initializing dispatcher: {str(e)}")
        sys.exit(1)
    
    # Load input data
    logger.info(f"Loading input data from {args.data}")
    print(f"\nLoading data from {args.data}")
    
    try:
        with open(args.data, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Successfully loaded {len(data)} samples")
        print(f"Loaded {len(data)} samples")
        
        # Validate data structure
        if data and isinstance(data, list):
            sample = data[0]
            required_keys = ['img_paths', 'question']
            missing_keys = [k for k in required_keys if k not in sample]
            if missing_keys:
                logger.warning(f"First sample missing keys: {missing_keys}")
        
    except FileNotFoundError:
        logger.error(f"Input file not found: {args.data}")
        print(f"\nError: Input file not found: {args.data}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in input file: {str(e)}")
        print(f"\nError: Invalid JSON in input file: {str(e)}")
        sys.exit(1)
    
    # Process data with checkpointing
    logger.info("Starting data processing with checkpointing")
    try:
        output_data = process_data_with_checkpointing(
            dispatcher, data, args.output, args.save_every
        )
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        print("\nProcessing interrupted. Progress has been saved.")
        print(f"Resume by running the same command again.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        print(f"\nError during processing: {str(e)}")
        print("Partial results may have been saved. Check the output file.")
        sys.exit(1)
    
    # Report final statistics
    print(f"\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"Results saved to: {args.output}")
    print(f"Total samples processed: {len(output_data)}")
    
    # Calculate success rate
    successful = sum(1 for s in output_data if not s.get('answer', '').startswith('Error:'))
    success_rate = (successful / len(output_data) * 100) if output_data else 0
    print(f"Success rate: {successful}/{len(output_data)} ({success_rate:.1f}%)")
    print("="*60)
    
    logger.info(f"Processing complete. {len(output_data)} samples processed")
    logger.info(f"Success rate: {success_rate:.1f}%")
    logger.info(f"Results saved to: {args.output}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}", exc_info=True)
        print(f"\nCritical error: {str(e)}")
        sys.exit(1)