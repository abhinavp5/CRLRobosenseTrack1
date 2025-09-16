MODEL_PATH="Qwen2.5-VL-32B-Instruct"
INPUT_DATA="robosense_track1_release_convert.json"
OUTPUT_DIR="outputs"
MAX_MODEL_LEN=10288
NUM_IMAGES_PER_PROMPT=6
PORT=8000

mkdir -p ${OUTPUT_DIR}

echo "Checking if vLLM server is ready..."
timeout=600  # 5+ minutes timeout
elapsed=0

until curl -s http://localhost:${PORT}/v1/models > /dev/null; do
    if [ $elapsed -ge $timeout ]; then
        echo "Timeout waiting for vLLM server to start after ${timeout} seconds"
        echo "Please check if service.sh is running and the server started correctly"
        exit 1
    fi
    echo "Waiting for vLLM to become ready... (${elapsed}s elapsed)"
    sleep 5
    elapsed=$((elapsed + 5))
done

echo "vLLM server is ready!"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/inference_results32b_${TIMESTAMP}.json"

echo "Running inference..."
python inference_enhancemcq.py \
    --model ${MODEL_PATH} \
    --data ${INPUT_DATA} \
    --output ${OUTPUT_FILE} \
    --max_model_len ${MAX_MODEL_LEN} \
    --num_images_per_prompt ${NUM_IMAGES_PER_PROMPT} \
    --api_base "http://localhost:${PORT}/v1" \
    --prompting_strategy selective_prompting \
    --mcq_self_consistency 1  #try 3 or 5
    
echo "inference completed $(date)"