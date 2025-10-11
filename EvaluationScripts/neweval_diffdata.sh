#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=4:00:00
#SBATCH --partition=h200_normal_q
#SBATCH --account=ece_6514
#SBATCH --gres=gpu:1

module load Miniconda3
module load CUDA/12.6.0

source activate myenv_fixed

export VLLM_WORKER_MULTIPROC_METHOD=spawn
NUM_GPUS=1
MODEL=/home/aaryankamdar/LLaMA-Factory/saves/qwen25_3b_instruct/lraaryanp12hrskamdar

MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.95,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

# Fix: Use absolute path for output directory
OUTPUT_DIR=evalresultsfinal
mkdir -p $OUTPUT_DIR

lighteval vllm $MODEL_ARGS \
    "lighteval|aime24|0|0,lighteval|aime25|0|0,lighteval|math_500|0|0,lighteval|gpqa:diamond|0|0" \
    --save-details \
    --output-dir $OUTPUT_DIR
