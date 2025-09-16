GPU_NUM=$1  

vllm serve Qwen2.5-VL-32B-Instruct \
  --trust-remote-code \
  --tensor-parallel-size $GPU_NUM \
  --max-model-len 12288 \
  --generation-config vllm \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 4 \
  --allowed-local-media-path / \
  --port 8000
 