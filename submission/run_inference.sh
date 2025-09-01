#!/bin/bash

# ============================================================================
# Unified VLM Inference Runner Script
# 
# This script starts the vLLM server and runs the inference pipeline
# on a system with 4x A100 GPUs
#
# Usage:
#   ./run_inference.sh <input_data.json> <output_results.json>
#
# Example:
#   ./run_inference.sh input_data.json results.json
#
# ============================================================================

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODEL="Qwen/Qwen2.5-VL-72B-Instruct"
TENSOR_PARALLEL_SIZE=4
MAX_MODEL_LEN=32768
GPU_MEMORY_UTILIZATION=0.9
PORT=8000
API_BASE="http://localhost:${PORT}/v1"

# Inference parameters
TEMPERATURE=0.2
TOP_P=0.2
MAX_TOKENS=1024
NUM_TEMPORAL_FRAMES=5
TEMPORAL_STRATEGY="sequential"
SAVE_EVERY=10

# Parse command line arguments
if [ $# -lt 2 ]; then
    echo -e "${RED}Error: Insufficient arguments${NC}"
    echo "Usage: $0 <input_data.json> <output_results.json> [model]"
    echo "Example: $0 input_data.json results.json"
    echo "         $0 input_data.json results.json Qwen/Qwen2.5-VL-32B-Instruct"
    exit 1
fi

INPUT_DATA="$1"
OUTPUT_FILE="$2"

# Optional: Override model if provided as third argument
if [ $# -ge 3 ]; then
    MODEL="$3"
fi

# Check if input file exists
if [ ! -f "$INPUT_DATA" ]; then
    echo -e "${RED}Error: Input file '$INPUT_DATA' not found${NC}"
    exit 1
fi

# Function to print separator
print_separator() {
    echo "================================================================"
}

# Function to check if vLLM server is running
check_vllm_server() {
    curl -s "${API_BASE}/models" > /dev/null 2>&1
    return $?
}

# Function to kill existing vLLM server
kill_vllm_server() {
    echo -e "${YELLOW}Checking for existing vLLM servers...${NC}"
    
    # Find and kill any process using port 8000
    if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${YELLOW}Found existing process on port ${PORT}, terminating...${NC}"
        lsof -Pi :${PORT} -sTCP:LISTEN -t | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    
    # Also kill any python processes running vllm
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
}

# Function to start vLLM server
start_vllm_server() {
    echo -e "${BLUE}Starting vLLM server...${NC}"
    print_separator
    echo "Configuration:"
    echo "  Model: ${MODEL}"
    echo "  GPUs: ${TENSOR_PARALLEL_SIZE}x A100"
    echo "  Max Model Length: ${MAX_MODEL_LEN}"
    echo "  GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}"
    echo "  Port: ${PORT}"
    print_separator
    
    # Prepare log file
    VLLM_LOG="vllm_server_$(date +%Y%m%d_%H%M%S).log"
    
    # Start vLLM server in background
    nohup python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
        --max-model-len ${MAX_MODEL_LEN} \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
        --port ${PORT} \
        --trust-remote-code \
        --max-num-seqs 32 \
        --enforce-eager \
        --image-input-type pixel_values \
        --image-token-id 151655 \
        --image-input-shape 1,3,448,448 \
        --image-feature-size 1176 \
        > "${VLLM_LOG}" 2>&1 &

    vllm serve "$MODEL" \
        --port ${PORTS[$IDX]} \
        --allowed-local-media-path / \
        --gpu-memory-utilization 0.95 \
        --trust-remote-code \
        --disable-log-requests \
        >logs/vllm_${IDX}.log 2>&1 &
    
    VLLM_PID=$!
    echo -e "${GREEN}vLLM server started with PID: ${VLLM_PID}${NC}"
    echo "Server log: ${VLLM_LOG}"
    
    # Wait for server to be ready
    echo -e "${YELLOW}Waiting for vLLM server to initialize...${NC}"
    MAX_WAIT=120  # Maximum wait time in seconds
    WAITED=0
    
    while ! check_vllm_server; do
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo -e "${RED}Error: vLLM server failed to start within ${MAX_WAIT} seconds${NC}"
            echo "Check the log file: ${VLLM_LOG}"
            tail -n 50 "${VLLM_LOG}"
            exit 1
        fi
        
        echo -n "."
        sleep 2
        WAITED=$((WAITED + 2))
    done
    
    echo ""
    echo -e "${GREEN}✓ vLLM server is ready!${NC}"
    print_separator
}

# Function to run inference
run_inference() {
    echo -e "${BLUE}Starting inference pipeline...${NC}"
    print_separator
    echo "Input: ${INPUT_DATA}"
    echo "Output: ${OUTPUT_FILE}"
    echo "Parameters:"
    echo "  Temperature: ${TEMPERATURE}"
    echo "  Top-p: ${TOP_P}"
    echo "  Max Tokens: ${MAX_TOKENS}"
    echo "  Temporal Frames: ${NUM_TEMPORAL_FRAMES}"
    echo "  Save Checkpoint Every: ${SAVE_EVERY} samples"
    print_separator
    
    # Run inference
    python inference.py \
        --model "${MODEL}" \
        --data "${INPUT_DATA}" \
        --output "${OUTPUT_FILE}" \
        --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
        --max_model_len ${MAX_MODEL_LEN} \
        --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
        --temperature ${TEMPERATURE} \
        --top_p ${TOP_P} \
        --max_tokens ${MAX_TOKENS} \
        --num_temporal_frames ${NUM_TEMPORAL_FRAMES} \
        --temporal_strategy ${TEMPORAL_STRATEGY} \
        --save_every ${SAVE_EVERY} \
        --api_base "${API_BASE}" \
        --log_level INFO
    
    INFERENCE_EXIT_CODE=$?
    
    if [ $INFERENCE_EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✓ Inference completed successfully!${NC}"
        echo "Results saved to: ${OUTPUT_FILE}"
    else
        echo -e "${RED}✗ Inference failed with exit code: ${INFERENCE_EXIT_CODE}${NC}"
    fi
    
    return $INFERENCE_EXIT_CODE
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"
    
    # Kill vLLM server if it's still running
    if [ ! -z "${VLLM_PID}" ]; then
        if kill -0 ${VLLM_PID} 2>/dev/null; then
            echo "Stopping vLLM server (PID: ${VLLM_PID})..."
            kill ${VLLM_PID} 2>/dev/null || true
            sleep 2
            
            # Force kill if still running
            if kill -0 ${VLLM_PID} 2>/dev/null; then
                kill -9 ${VLLM_PID} 2>/dev/null || true
            fi
        fi
    fi
    
    # Also try to kill by port
    kill_vllm_server
    
    echo -e "${GREEN}Cleanup complete${NC}"
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Main execution
main() {
    echo ""
    print_separator
    echo -e "${BLUE}RoboSense VLM Inference Pipeline${NC}"
    echo "System: 4x NVIDIA A100 GPUs"
    print_separator
    echo ""
    
    # Check CUDA availability
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}Error: nvidia-smi not found. Please ensure CUDA is installed.${NC}"
        exit 1
    fi
    
    # Display GPU information
    echo -e "${BLUE}GPU Information:${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | head -4
    echo ""
    
    # Check Python and required packages
    echo -e "${BLUE}Checking dependencies...${NC}"
    
    if ! python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null; then
        echo -e "${RED}Error: PyTorch not installed${NC}"
        exit 1
    fi
    
    if ! python -c "import vllm; print(f'vLLM: {vllm.__version__}')" 2>/dev/null; then
        echo -e "${RED}Error: vLLM not installed${NC}"
        exit 1
    fi
    
    if ! python -c "from openai import OpenAI; print('OpenAI client: OK')" 2>/dev/null; then
        echo -e "${RED}Error: OpenAI Python client not installed${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ All dependencies satisfied${NC}"
    echo ""
    
    # Kill any existing vLLM server
    kill_vllm_server
    
    # Start vLLM server
    start_vllm_server
    
    # Run inference
    run_inference
    RESULT=$?
    
    # Exit with inference result code
    exit $RESULT
}

# Run main function
main