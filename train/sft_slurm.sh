#!/bin/bash
#SBATCH --account=PAS3098
#SBATCH --job-name=gemma3_sft
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=200G
#SBATCH --time=5:30:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# Load required modules
module load miniconda3/24.1.2-py310

# Activate your conda environment
conda activate note_curation

# Change to your working directory
cd /users/PAS2997/jwang580/bmt_train

# Set up NCCL environment variables for multi-node training
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export NCCL_TIMEOUT=1800
export TORCH_FSDP_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo

# Set up master node coordination
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Number of nodes: $SLURM_NNODES"
echo "Total tasks: $SLURM_NTASKS"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Job ID: $SLURM_JOB_ID"

# Training parameters
uid="$(date +%Y%m%d_%H%M%S)"
base_model="gghfez/gemma-3-27b-novision"
lr=1e-5
min_lr=0
epochs=5
weight_decay=1e-4
micro_batch_size=1 
gradient_accumulation_steps=1
max_steps=-1
push_to_hub=false

# Launch training with srun directly
srun python train/sft.py \
    --block_size=32768 \
    --max_length=32768 \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --train_file_path="results/datasets/combined_s1_clinical_cleaned_shuffled.parquet" \
    --model_name=${base_model} \
    --warmup_ratio=0.05 \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="train/fsdp_config_gemma.json" \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="ckpts/s1-${uid}" \
    --push_to_hub=${push_to_hub} \
    --report_to="wandb" \
    --save_only_model=True \
    --gradient_checkpointing=True \
    --accelerator_config='{"gradient_accumulation_kwargs": {"sync_each_batch": true}}' 