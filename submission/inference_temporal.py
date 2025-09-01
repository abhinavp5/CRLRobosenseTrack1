#!/usr/bin/env python3
"""
Temporal Vision-Language Model (VLM) Inference for Autonomous Driving

This module provides a comprehensive system for running multi-GPU inference on temporal
sequences of multi-camera images using Qwen2.5-VL models. It's designed specifically
for autonomous driving scenarios where understanding temporal context across multiple
camera views is crucial for scene understanding and decision making.

Key Features:
    - Temporal sequence processing across multiple camera views (6 cameras)
    - Multi-GPU support via tensor parallelism using vLLM
    - Fault-tolerant processing with automatic checkpointing
    - Flexible temporal strategies (sequential vs interleaved)
    - Production-ready logging and monitoring

Typical Usage:
    python inference.py \
        --model Qwen/Qwen2.5-VL-72B-Instruct \
        --data input_data.json \
        --output results.json \
        --num_temporal_frames 5 \
        --tensor_parallel_size 4

Author: RoboSense AI Team
Version: 1.0.0
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from openai import OpenAI
from collections import OrderedDict
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'inference_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parse command-line arguments for the temporal VLM inference system.
    
    This function defines all configurable parameters for the inference pipeline,
    including model selection, data paths, temporal processing settings, GPU
    configuration, and inference hyperparameters.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments containing:
            - model: Model identifier for Qwen2.5-VL variants
            - data: Path to input JSON data file
            - output: Path for saving inference results
            - num_temporal_frames: Number of temporal frames to process
            - temporal_strategy: How to arrange frames (sequential/interleaved)
            - tensor_parallel_size: Number of GPUs for parallelization
            - max_model_len: Maximum sequence length for the model
            - gpu_memory_utilization: Fraction of GPU memory to use
            - temperature: Sampling temperature for generation
            - top_p: Nucleus sampling parameter
            - max_tokens: Maximum tokens to generate per response
            - api_base: vLLM server endpoint URL
            - batch_size: Number of samples to process together
            - save_every: Checkpoint frequency
            - log_level: Logging verbosity level
    
    Example:
        >>> args = parse_arguments()
        >>> print(f"Using model: {args.model}")
        >>> print(f"Processing {args.num_temporal_frames} frames")
    """
    logger.debug("Parsing command-line arguments")
    parser = argparse.ArgumentParser(description='Temporal VLM Multi-GPU Inference for Qwen2.5-VL')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-VL-72B-Instruct',
                        help='Model name (Qwen2.5-VL-32B or Qwen2.5-VL-72B)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to input data JSON file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output JSON file')
    
    # Temporal settings
    parser.add_argument('--num_temporal_frames', type=int, default=5,
                        help='Number of temporal frames to use (including current)')
    parser.add_argument('--temporal_strategy', type=str, default='sequential',
                        choices=['sequential', 'interleaved'],
                        help='How to arrange temporal frames in prompt')
    
    # GPU and model settings
    parser.add_argument('--tensor_parallel_size', type=int, default=4,
                        help='Number of GPUs for tensor parallelism')
    parser.add_argument('--max_model_len', type=int, default=32768,
                        help='Maximum model length (increased for temporal frames)')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                        help='GPU memory utilization for vLLM')
    
    # Hyperparameters
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
    
    # Log parsed arguments
    logger.info("Command-line arguments parsed successfully")
    logger.debug(f"Arguments: {vars(args)}")
    
    return args

# System prompt for the VLM model - defines the task context and expectations
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

class TemporalVLMInference:
    """
    Main inference engine for temporal Vision-Language Model processing.
    
    This class handles the core functionality of processing temporal sequences
    of multi-camera images through a VLM model. It manages API connections,
    formats temporal prompts, and handles the inference pipeline.
    
    Attributes:
        client (OpenAI): OpenAI-compatible API client for vLLM server
        model (str): Model identifier for inference
        temperature (float): Sampling temperature (0.0-1.0)
        top_p (float): Nucleus sampling parameter
        max_tokens (int): Maximum generation length
        num_temporal_frames (int): Number of temporal frames to process
        temporal_strategy (str): Frame arrangement strategy
    
    Example:
        >>> vlm = TemporalVLMInference(
        ...     model_name="Qwen/Qwen2.5-VL-72B-Instruct",
        ...     api_base="http://localhost:8000/v1",
        ...     temperature=0.2,
        ...     top_p=0.2,
        ...     max_tokens=1024,
        ...     num_temporal_frames=5,
        ...     temporal_strategy="sequential"
        ... )
        >>> result = vlm.process_sample(sample_data)
    """
    
    def __init__(self, model_name: str, api_base: str, temperature: float, 
                 top_p: float, max_tokens: int, num_temporal_frames: int,
                 temporal_strategy: str):
        """
        Initialize the temporal VLM inference engine.
        
        Args:
            model_name: Identifier for the VLM model to use
            api_base: Base URL for the vLLM API server
            temperature: Sampling temperature for generation (0.0-1.0)
            top_p: Top-p nucleus sampling parameter (0.0-1.0)
            max_tokens: Maximum number of tokens to generate
            num_temporal_frames: Number of temporal frames to include
            temporal_strategy: Strategy for arranging temporal frames
                              ('sequential' or 'interleaved')
        
        Raises:
            ValueError: If temporal_strategy is not 'sequential' or 'interleaved'
        """
        logger.info(f"Initializing TemporalVLMInference with model: {model_name}")
        logger.debug(f"API base: {api_base}")
        logger.debug(f"Temperature: {temperature}, Top-p: {top_p}, Max tokens: {max_tokens}")
        logger.debug(f"Temporal frames: {num_temporal_frames}, Strategy: {temporal_strategy}")

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
        """
        Extract temporal sequence of frames from sample data.
        
        This method processes the input sample to extract a temporal sequence of
        camera frames. It selects the most recent N-1 history frames and combines
        them with the current frame to create a temporal context window.
        
        Think of this like collecting frames from a flipbook - we need them in order
        to understand the motion when we flip through them.
        
        Args:
            sample: Input data dictionary containing:
                - 'history_frames': Dictionary of historical frame data
                - 'img_paths': Current frame image paths
        
        Returns:
            List of dictionaries, each containing camera view paths for a single
            temporal frame. Frames are ordered from oldest to newest.
        
        Example:
            >>> frames = vlm.extract_temporal_frames(sample)
            >>> print(f"Extracted {len(frames)} temporal frames")
            >>> print(f"First frame cameras: {frames[0].keys()}")
        """
        logger.debug(f"Extracting temporal frames for sample")
        temporal_frames = []
        
        # Get history frames (oldest to newest)
        history_frames = sample.get('history_frames', {})
        history_tokens = list(history_frames.keys())
        logger.debug(f"Found {len(history_tokens)} history frames available")
        
        # Sort by timestamp if available, or use as-is
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
        """
        Format temporal frames into a structured prompt for the VLM.
        
        This method arranges temporal frames according to the specified strategy:
        - Sequential: Groups all camera views by temporal frame
        - Interleaved: Groups temporal frames by camera view
        
        Like arranging multiple photos on a table - we can either group by time
        (all views at T1, then all at T2) or by camera (front view across time, 
        then right view across time).
        
        Args:
            frames: List of frame dictionaries containing camera view paths
            question: The question to answer about the temporal sequence
        
        Returns:
            List of content dictionaries formatted for the OpenAI API, containing
            text labels and image URLs in the appropriate order.
        
        Example:
            >>> content = vlm.format_temporal_prompt(frames, "What vehicles are approaching?")
            >>> print(f"Generated {len(content)} content items")
        """
        logger.info(f"Formatting temporal prompt with strategy: {self.temporal_strategy}")
        logger.debug(f"Processing {len(frames)} frames with question: {question[:50]}...")
        content = []
        
        if self.temporal_strategy == 'sequential':
            # Sequential strategy: Group by temporal frame
            # Frame 1 (all views), Frame 2 (all views), etc.
            logger.debug("Using sequential temporal strategy")
            
            for frame_idx, frame_imgs in enumerate(frames, 1):
                # Add frame header
                content.append({
                    "type": "text",
                    "text": f"\n--- Frame {frame_idx} of {len(frames)} ---"
                })
                logger.debug(f"Processing frame {frame_idx}/{len(frames)}")
                
                # Add all camera views for this temporal frame
                # Process cameras in a consistent order for spatial coherence
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
            # Front (T1, T2, ...), Right (T1, T2, ...), etc.
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
        """
        Process a single sample through the temporal VLM pipeline.
        
        This method orchestrates the complete inference process for a single sample:
        1. Extracts temporal frames from the sample data
        2. Formats the prompt according to the temporal strategy
        3. Sends the request to the VLM API with retry logic
        4. Returns the generated response
        
        Args:
            sample: Input data dictionary containing:
                - 'history_frames': Historical frame data
                - 'img_paths': Current frame paths
                - 'question': Question to answer about the scene
        
        Returns:
            Generated response string from the VLM model, or an error message
            if processing fails.
        
        Raises:
            Exception: If all retry attempts fail
        
        Example:
            >>> sample = {
            ...     'history_frames': {...},
            ...     'img_paths': {...},
            ...     'question': "What vehicles are visible?"
            ... }
            >>> response = vlm.process_sample(sample)
            >>> print(response)
        """
        sample_id = sample.get('frame_token', sample.get('id', 'unknown'))
        logger.info(f"Processing sample: {sample_id}")
        start_time = time.time()
        
        try:
            # Step 1: Extract temporal frames from the sample
            logger.debug("Extracting temporal frames")
            temporal_frames = self.extract_temporal_frames(sample)
            
            # Step 2: Prepare messages with system prompt
            logger.debug("Preparing messages with system prompt")
            messages = [
                {"role": "system", "content": TEMPORAL_SYSTEM_PROMPT.format(
                    num_frames=len(temporal_frames)
                )}
            ]
            
            # Step 3: Format temporal content according to strategy
            logger.debug("Formatting temporal content")
            content = self.format_temporal_prompt(temporal_frames, sample['question'])
            
            # Add user message with formatted content
            messages.append({
                "role": "user",
                "content": content
            })
            logger.debug(f"Prepared message with {len(content)} content items")
            
            # Step 4: Call API with retry logic for robustness
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
            logger.error(error_msg, exc_info=True)  # Include stack trace in log
            return error_msg

def setup_vllm_server(args):
    """
    Setup and verify vLLM server configuration for multi-GPU inference.
    
    This function checks if a vLLM server is already running, and if not,
    provides the command to start one with the appropriate configuration
    for tensor-parallel multi-GPU inference.
    
    This is like setting up a team of workers (GPUs) to collaborate on a big task.
    Each GPU handles a portion of the model, working together seamlessly.
    
    Args:
        args: Command-line arguments containing:
            - model: Model identifier
            - api_base: vLLM server URL
            - tensor_parallel_size: Number of GPUs
            - max_model_len: Maximum sequence length
            - gpu_memory_utilization: GPU memory fraction
    
    Returns:
        None. Prints instructions if server needs to be started.
    
    Note:
        The function will detect if a vLLM server is already running by
        attempting to connect to it. If not running, it provides the
        exact command needed to start the server.
    """
    import subprocess
    
    logger.info("Checking vLLM server status...")
    
    # Check if vLLM server is already running by attempting connection
    try:
        logger.debug(f"Attempting to connect to vLLM server at {args.api_base}")
        test_client = OpenAI(api_key="EMPTY", base_url=args.api_base)
        test_client.models.list()
        logger.info("✓ vLLM server is already running and accessible")
        return
    except Exception as e:
        logger.info("vLLM server not detected, preparing launch command")
        logger.debug(f"Connection error: {str(e)}")
    
    # Prepare vLLM launch command with all necessary parameters
    vllm_cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--port", "8000",
        "--trust-remote-code",  # Required for Qwen models to load custom code
        "--max-num-seqs", "32",  # Maximum concurrent sequences
        "--enforce-eager",  # Disable CUDA graphs for better memory efficiency
    ]
    
    logger.debug(f"Base vLLM command prepared with {args.tensor_parallel_size} GPUs")
    
    # Add Qwen2.5-VL specific settings for vision processing
    if "Qwen2.5-VL" in args.model:
        logger.debug("Adding Qwen2.5-VL specific vision parameters")
        vllm_cmd.extend([
            "--image-input-type", "pixel_values",
            "--image-token-id", "151655",  # Qwen2.5-VL special token for images
            "--image-input-shape", "1,3,448,448",  # Image input dimensions
            "--image-feature-size", "1176",  # Vision encoder output size
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
    print("3. Wait for the server to fully initialize (look for 'Uvicorn running on...')")
    print("4. Run this script again to process your data")
    print("="*70 + "\n")
    
    logger.info("Exiting - vLLM server needs to be started first")
    return

def process_data_with_checkpointing(vlm: TemporalVLMInference, data: List[Dict[str, Any]], 
                                   output_path: str, save_every: int = 10) -> List[Dict[str, Any]]:
    """
    Process data with periodic checkpointing for fault tolerance.
    
    This function implements a robust data processing pipeline with automatic
    checkpointing and resume capabilities. It can recover from interruptions
    and continue processing from where it left off.
    
    Args:
        vlm: TemporalVLMInference instance for processing
        data: List of sample dictionaries to process
        output_path: Path to save results and checkpoints
        save_every: Number of samples between checkpoint saves
    
    Returns:
        List of processed samples with inference results added
    
    Features:
        - Automatic resume from previous runs
        - Periodic checkpointing for fault tolerance
        - Progress tracking with tqdm
        - Atomic file writes to prevent corruption
    
    Example:
        >>> results = process_data_with_checkpointing(
        ...     vlm, data, "results.json", save_every=10
        ... )
        >>> print(f"Processed {len(results)} samples")
    """
    logger.info(f"Starting data processing with checkpointing (save_every={save_every})")
    
    # Load existing results if any (for resuming interrupted runs)
    if os.path.exists(output_path):
        logger.info(f"Found existing output file: {output_path}")
        with open(output_path, 'r') as f:
            output_data = json.load(f)
        
        # Track which samples have already been processed
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
            # Ensure sample has an ID for tracking
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
    with tqdm(total=len(remaining_data), desc="Processing temporal sequences") as pbar:
        for idx, sample in enumerate(remaining_data):
            sample_id = sample.get('id', f"sample_{idx}")
            logger.debug(f"Processing sample {idx+1}/{len(remaining_data)}: {sample_id}")
            # Process sample through VLM pipeline
            try:

                # Something going wrong here 
                answer = vlm.process_sample(sample)
            except Exception as e:
                
                logger.error(f"Failed to process sample {sample_id}: {str(e)}")
                answer = f"Error: {str(e)}"
            
            # Add result to output data
            output_sample = sample.copy()
            output_sample['answer'] = answer
            output_sample['processed_at'] = datetime.now().isoformat()  # Add timestamp
            output_data.append(output_sample)
            
            logger.debug(f"Sample {sample_id} processed successfully")
            
            # Save checkpoint periodically for fault tolerance
            if (idx + 1) % save_every == 0:
                save_checkpoint(output_path, output_data)
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                logger.info(f"Checkpoint saved at sample {idx + 1} ({rate:.2f} samples/sec)")
                print(f"\nCheckpoint saved at sample {idx + 1} ({rate:.2f} samples/sec)")
            
            # Update progress bar
            pbar.update(1)
    
    # Final save of all results
    save_checkpoint(output_path, output_data)
    
    # Log processing statistics
    total_time = time.time() - start_time
    avg_time = total_time / len(remaining_data) if remaining_data else 0
    logger.info(f"Processing complete: {len(remaining_data)} samples in {total_time:.2f}s")
    logger.info(f"Average processing time: {avg_time:.2f}s per sample")
    
    return output_data

def save_checkpoint(output_path: str, data: List[Dict[str, Any]]):
    """
    Save data with atomic write operation for safety.
    
    This function uses a temporary file and atomic rename to ensure that
    the output file is never left in a corrupted state, even if the process
    is interrupted during writing.
    
    Args:
        output_path: Target path for the output file
        data: List of data dictionaries to save
    
    Note:
        Uses os.replace() for atomic file replacement, which ensures
        the operation is atomic on POSIX systems.
    """
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
        # Clean up temporary file if it exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

def main():
    """
    Main entry point for the temporal VLM inference system.
    
    This function orchestrates the entire inference pipeline:
    1. Parses command-line arguments
    2. Validates GPU availability
    3. Sets up or verifies vLLM server
    4. Initializes the inference engine
    5. Loads input data
    6. Processes samples with checkpointing
    7. Reports final statistics
    
    The function is designed to be idempotent and can resume from
    previous runs if interrupted.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    logger.info("="*60)
    logger.info("Temporal VLM Inference System Starting")
    logger.info("="*60)
    
    # Display configuration summary
    print("\n" + "=" * 60)
    print("Temporal Multi-GPU Inference Configuration")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Temporal Frames: {args.num_temporal_frames}")
    print(f"Temporal Strategy: {args.temporal_strategy}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size} GPUs")
    print(f"Max Model Length: {args.max_model_len}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Max Tokens: {args.max_tokens}")
    print("=" * 60)
    
    # Log full configuration for debugging
    logger.debug(f"Full configuration: {vars(args)}")
    
    # Check GPU availability and log details
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"CUDA available with {num_gpus} GPU(s)")
        print(f"\nDetected {num_gpus} GPUs:")
        
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9  # Convert to GB
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            logger.debug(f"GPU {i}: {gpu_name}, Memory: {gpu_memory:.1f} GB")
        
        # Validate tensor parallel size
        if args.tensor_parallel_size > num_gpus:
            logger.warning(f"Tensor parallel size ({args.tensor_parallel_size}) exceeds available GPUs ({num_gpus})")
            print(f"\nWarning: Requested {args.tensor_parallel_size} GPUs but only {num_gpus} available!")
    else:
        logger.error("No CUDA GPUs detected - this will likely fail")
        print("\nError: No GPUs detected! This system requires CUDA-capable GPUs.")
        sys.exit(1)
    
    # Setup or verify vLLM server
    logger.info("Setting up vLLM server")
    setup_vllm_server(args)
    
    # Initialize temporal VLM inference engine
    logger.info("Initializing temporal VLM inference engine")
    try:
        vlm = TemporalVLMInference(
            model_name=args.model,
            api_base=args.api_base,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            num_temporal_frames=args.num_temporal_frames,
            temporal_strategy=args.temporal_strategy
        )
        logger.info("VLM inference engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize VLM: {str(e)}")
        print(f"\nError initializing VLM: {str(e)}")
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
    
    # Process data with checkpointing and fault tolerance
    logger.info("Starting data processing with checkpointing")
    try:
        output_data = process_data_with_checkpointing(
            vlm, data, args.output, args.save_every
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
    
    # Calculate and display success rate
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