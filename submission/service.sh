# Set cache to scratch directory
export PATH="$HOME/.local/bin:$PATH"

export HF_HOME=/scratch/$USER/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/$USER/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/$USER/huggingface/transformers

# Create the directories
mkdir -p /scratch/$USER/huggingface/hub

vllm serve "Qwen/Qwen2.5-VL-72B-Instruct" \
    --allowed-local-media-path / \
    --tensor-parallel-size 4 \
    --enforce-eager

# export PATH="$HOME/.local/bin:$PATH"
# pip install "numpy>=1.25.2,<2.0.0" --upgrade --force-reinstall