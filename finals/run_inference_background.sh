#!/bin/bash

# ============================================================================

#   ./run_inference_background.sh start              # Start vLLM server
#   ./run_inference_background.sh stop               # Stop vLLM server
#   ./run_inference_background.sh status             # Check server status
#   ./run_inference_background.sh run <input> <output>  # Run inference only
#   ./run_inference_background.sh full <input> <output> # Start server & run
#
# ============================================================================

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
MODEL="Qwen/Qwen2.5-VL-72B-Instruct"
TENSOR_PARALLEL_SIZE=4
MAX_MODEL_LEN=32768
GPU_MEMORY_UTILIZATION=0.9
PORT=8000
API_BASE="http://localhost:${PORT}/v1"
VLLM_PID_FILE="/tmp/vllm_server.pid"
VLLM_LOG_FILE="/tmp/vllm_server.log"

# Inference parameters
TEMPERATURE=0.2
TOP_P=0.2
MAX_TOKENS=1024
NUM_TEMPORAL_FRAMES=5
TEMPORAL_STRATEGY="sequential"
SAVE_EVERY=10

# Function to check if server is running
is_server_running() {
    if [ -f "$VLLM_PID_FILE" ]; then
        PID=$(cat "$VLLM_PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# Function to check server health
check_server_health() {
    curl -s "${API_BASE}/models" > /dev/null 2>&1
    return $?
}

# Start vLLM server
start_server() {
    echo -e "${BLUE}Starting vLLM server...${NC}"
    
    if is_server_running; then
        echo -e "${YELLOW}vLLM server is already running (PID: $(cat $VLLM_PID_FILE))${NC}"
        return 0
    fi
    
    # Clean up old PID file
    rm -f "$VLLM_PID_FILE"
    
    # Start server
    echo "Starting vLLM server with ${TENSOR_PARALLEL_SIZE}x A100 GPUs..."
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
        > "${VLLM_LOG_FILE}" 2>&1 &
    
    PID=$!
    echo $PID > "$VLLM_PID_FILE"
    
    echo "Server started with PID: $PID"
    echo "Waiting for initialization..."
    
    # Wait for server to be ready
    MAX_WAIT=120
    WAITED=0
    
    while ! check_server_health; do
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo -e "${RED}Server failed to start within ${MAX_WAIT} seconds${NC}"
            echo "Last 20 lines of log:"
            tail -n 20 "$VLLM_LOG_FILE"
            stop_server
            exit 1
        fi
        
        echo -n "."
        sleep 2
        WAITED=$((WAITED + 2))
    done
    
    echo ""
    echo -e "${GREEN}✓ vLLM server is ready!${NC}"
    echo "Server log: $VLLM_LOG_FILE"
}

# Stop vLLM server
stop_server() {
    echo -e "${BLUE}Stopping vLLM server...${NC}"
    
    if [ -f "$VLLM_PID_FILE" ]; then
        PID=$(cat "$VLLM_PID_FILE")
        
        if kill -0 "$PID" 2>/dev/null; then
            echo "Stopping server (PID: $PID)..."
            kill "$PID"
            
            # Wait for graceful shutdown
            sleep 3
            
            # Force kill if still running
            if kill -0 "$PID" 2>/dev/null; then
                echo "Force killing server..."
                kill -9 "$PID" 2>/dev/null || true
            fi
            
            echo -e "${GREEN}Server stopped${NC}"
        else
            echo -e "${YELLOW}Server process not found${NC}"
        fi
        
        rm -f "$VLLM_PID_FILE"
    else
        echo -e "${YELLOW}No server PID file found${NC}"
    fi
    
    # Also kill any orphaned processes
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
}

# Check server status
check_status() {
    echo -e "${BLUE}Checking vLLM server status...${NC}"
    echo "================================"
    
    if is_server_running; then
        PID=$(cat "$VLLM_PID_FILE")
        echo -e "Status: ${GREEN}RUNNING${NC}"
        echo "PID: $PID"
        
        if check_server_health; then
            echo -e "Health: ${GREEN}HEALTHY${NC}"
            echo "API Endpoint: $API_BASE"
            
            # Show model info
            echo ""
            echo "Model Information:"
            curl -s "${API_BASE}/models" | python -m json.tool 2>/dev/null || echo "Could not fetch model info"
        else
            echo -e "Health: ${RED}UNHEALTHY${NC}"
            echo "Server is running but not responding to API calls"
        fi
        
        # Show resource usage
        echo ""
        echo "Resource Usage:"
        ps -p $PID -o pid,vsz,rss,pcpu,pmem,comm --no-headers || true
        
    else
        echo -e "Status: ${RED}NOT RUNNING${NC}"
    fi
    
    echo ""
    echo "Log file: $VLLM_LOG_FILE"
    
    if [ -f "$VLLM_LOG_FILE" ]; then
        echo "Last 5 log lines:"
        tail -n 5 "$VLLM_LOG_FILE"
    fi
}

# Run inference
run_inference() {
    INPUT_DATA="$1"
    OUTPUT_FILE="$2"
    
    if [ -z "$INPUT_DATA" ] || [ -z "$OUTPUT_FILE" ]; then
        echo -e "${RED}Error: Missing arguments${NC}"
        echo "Usage: $0 run <input_data.json> <output_file.json>"
        exit 1
    fi
    
    if [ ! -f "$INPUT_DATA" ]; then
        echo -e "${RED}Error: Input file '$INPUT_DATA' not found${NC}"
        exit 1
    fi
    
    # Check if server is running
    if ! is_server_running || ! check_server_health; then
        echo -e "${YELLOW}vLLM server is not running or unhealthy${NC}"
        echo "Starting server first..."
        start_server
    fi
    
    echo -e "${BLUE}Running inference...${NC}"
    echo "Input: $INPUT_DATA"
    echo "Output: $OUTPUT_FILE"
    echo ""
    
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
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Inference completed successfully!${NC}"
        echo "Results: $OUTPUT_FILE"
    else
        echo -e "${RED}✗ Inference failed${NC}"
        exit 1
    fi
}

# Show usage
show_usage() {
    echo "Usage: $0 {start|stop|status|restart|run|full|logs|help}"
    echo ""
    echo "Commands:"
    echo "  start                - Start vLLM server in background"
    echo "  stop                 - Stop vLLM server"
    echo "  status               - Check server status"
    echo "  restart              - Restart vLLM server"
    echo "  run <input> <output> - Run inference (start server if needed)"
    echo "  full <input> <output>- Stop, start server, then run inference"
    echo "  logs [n]            - Show last n lines of server log (default: 50)"
    echo "  help                - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start"
    echo "  $0 run input_data.json results.json"
    echo "  $0 full input_data.json results.json"
    echo "  $0 logs 100"
}

# Show logs
show_logs() {
    LINES=${1:-50}
    
    if [ -f "$VLLM_LOG_FILE" ]; then
        echo -e "${BLUE}Last $LINES lines of vLLM server log:${NC}"
        echo "================================"
        tail -n "$LINES" "$VLLM_LOG_FILE"
    else
        echo -e "${YELLOW}No log file found${NC}"
    fi
}

# Main command handler
case "$1" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    status)
        check_status
        ;;
    restart)
        stop_server
        sleep 2
        start_server
        ;;
    run)
        run_inference "$2" "$3"
        ;;
    full)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo -e "${RED}Error: Missing arguments${NC}"
            echo "Usage: $0 full <input_data.json> <output_file.json>"
            exit 1
        fi
        stop_server
        sleep 2
        start_server
        run_inference "$2" "$3"
        ;;
    logs)
        show_logs "$2"
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo -e "${RED}Error: Unknown command '$1'${NC}"
        echo ""
        show_usage
        exit 1
        ;;
esac