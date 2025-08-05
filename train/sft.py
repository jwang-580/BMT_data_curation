# SFT using huggingface TRL

import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, concatenate_datasets, DatasetDict
import transformers
from transformers import Gemma3ForConditionalGeneration
from accelerate.state import AcceleratorState
import trl
from trl import DataCollatorForCompletionOnlyLM
import dotenv
import wandb
import torch

dotenv.load_dotenv()

def is_main_process() -> bool:
    """
    Safe check that works with or without torch.distributed initialised.
    Returns True on the single-GPU case or on global rank 0.
    """
    return (not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0)

if is_main_process():
    wandb.login(key=os.getenv("WANDB_API_KEY"))
else:
    os.environ.update({
        "WANDB_MODE": "disabled",
        "WANDB_SILENT": "true",
    })


@dataclass
class TrainingConfig:
    model_name: str = field(default="gghfez/gemma-3-27b-novision")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="gemma-3-bmt-sft")
    wandb_entity: Optional[str] = field(default="osu_gpt")
    train_file_path: Optional[str] = field(default='results/datasets/combined_s1_clinical_cleaned_shuffled.parquet') 
    dagger: bool = field(default=False)

    def __post_init__(self):
        if self.wandb_project:
            os.environ['WANDB_PROJECT'] = self.wandb_project
        if self.wandb_entity:
            os.environ['WANDB_ENTITY'] = self.wandb_entity

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model
    kwargs = {"torch_dtype": "auto", "attn_implementation": "flash_attention_2"}
    # model = transformers.AutoModelForImageTextToText.from_pretrained(config.model_name, token=os.getenv("HUGGINGFACE_HUB_TOKEN"), **kwargs)
    model = transformers.Gemma3ForCausalLM.from_pretrained(config.model_name, token=os.getenv("HUGGINGFACE_HUB_TOKEN"), **kwargs, low_cpu_mem_usage=True)
    
    # Move model to GPU after distributed initialization has set the device
    if torch.cuda.is_available():
        model = model.to(torch.cuda.current_device())
        print(f"Model moved to GPU: {torch.cuda.current_device()}")

    dataset = load_dataset("parquet", data_files=config.train_file_path)

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    
    response_template = "<start_of_turn>model\n"
    tokenizer.pad_token = "<pad>"
    
    # Create the completion-only data collator
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
   
    # DataCollatorForCompletionOnlyLM has a bug for instruction/response templates if no context is provided, see https://huggingface.co/docs/trl/v0.7.2/en/sft_trainer#using-tokenids-directly-for-responsetemplate

    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size
    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'], # no test set
        data_collator=data_collator,
        args=args
    )

    # Debug: show how Accelerate wrapped the model/FSDP status
    state = AcceleratorState()
    print(">>> Distributed type :", state.distributed_type)
    if hasattr(state, 'fsdp_plugin') and state.fsdp_plugin is not None and hasattr(state.fsdp_plugin, "fsdp_kwargs"):
        print(">>> FSDP options :", state.fsdp_plugin.fsdp_kwargs)
    else:
        print(">>> FSDP plugin not available or not configured")
    print(">>> Model wrapper    :", type(trainer.model))

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


def init_distributed_slurm():
    """Initialize distributed training using SLURM environment variables"""
    
    # Get SLURM environment variables
    world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    ntasks_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", "1"))
    
    # Calculate local rank correctly: rank within the node (0-3 for each node)
    local_rank = rank % ntasks_per_node
    
    # Set standard distributed training environment variables for accelerate
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    
    # Get master address from environment (set by SLURM script)
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    
    print(f"Initializing distributed training: rank {rank}/{world_size}, local_rank {local_rank}")
    print(f"Master: {master_addr}:{master_port}")
    
    # Set the GPU device based on local rank
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.cuda.current_device()
        print(f"Process {rank} using GPU {device}: {torch.cuda.get_device_name(device)}")
    
    # Initialize the process group
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank
    )

if __name__ == "__main__":
    init_distributed_slurm()
    train()
    torch.distributed.destroy_process_group()