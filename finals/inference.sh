export PATH="$HOME/.local/bin:$PATH"

MODEL_PATH="Qwen/Qwen2.5-VL-72B-Instruct"
INPUT_DATA="dataset_format_temporal.json"
OUTPUT_DIR="outputs"
MAX_MODEL_LEN=32768
NUM_IMAGES_PER_PROMPT=6
TEMPERATURE=0.2
TOP_P=0.2
MAX_TOKENS=512
PORT=8000

mkdir -p ${OUTPUT_DIR}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/inference_results_${TIMESTAMP}.json"

echo "Running inference..."
# python inference.py \
#     --model ${MODEL_PATH} \
#     --data ${INPUT_DATA} \
#     --output ${OUTPUT_FILE} \
#     --max_model_len ${MAX_MODEL_LEN} \
#     --num_images_per_prompt ${NUM_IMAGES_PER_PROMPT} \
#     --temperature ${TEMPERATURE} \
#     --top_p ${TOP_P} \
#     --max_tokens ${MAX_TOKENS} \
#     --api_base "http://localhost:${PORT}/v1"

# python inference.py \
#     --model Qwen/Qwen2.5-VL-72B-Instruct \
#     --data dataset_format_temporal.json \
#     --output ${OUTPUT_FILE} \
#     --num_temporal_frames 5 \
#     --tensor_parallel_size 4

python inference.py \
    --model $MODEL_PATH \
    --data $INPUT_DATA \
    --output $OUTPUT_FILE \
    --num_temporal_frames 5 \
    --temporal_strategy sequential \
    --tensor_parallel_size 4 \
    --max_model_len $MAX_MODEL_LEN \
    --gpu_memory_utilization 0.9 \
    --max_tokens 1024 \
    --api_base http://localhost:$PORT/v1 \