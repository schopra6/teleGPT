import torch
from contextlib import nullcontext
from dataclasses import dataclass
import os

@dataclass
class GPTConfig:

    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    block_size: int = 1024
    bias: bool = True

@dataclass
class IOMetricsConfig:

    out_dir: str = 'out'
    eval_interval: int = 3
    log_interval: int = 1
    eval_only: bool = False
    save_interval: int = 90
    always_save_checkpoint: bool = True
    init_from: str = 'scratch'
    log: bool = True
    name: str = 'gpt2'
    folder: str = '../../out/'
    dataset: str = '../dataset/'
    file: str = None

@dataclass
class DataConfig:

    gradient_accumulation_steps: int = 5 * torch.cuda.device_count() if torch.cuda.is_available() else 5
    batch_size: int = 12

@dataclass
class OptimizerConfig:

    max_iters: int = 50
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    betas: tuple = (beta1, beta2)
    grad_clip: float = 1.0

@dataclass
class LearningRateConfig:

    learning_rate: float = 6e-4
    decay_lr: bool = True
    min_lr: float = 6e-5

@dataclass
class DDPConfig:


    backend: str = 'nccl'
    ddp: bool = int(os.environ.get('RANK', -1)) != -1

@dataclass
class SystemConfig:


    use_cuda: bool = torch.cuda.is_available()
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    compile: bool = True
    num_workers: int = int(os.environ['SLURM_CPUS_PER_TASK']) if 'SLURM_CPUS_PER_TASK' in os.environ else 4
    is_slurm: bool = os.getenv('SLURM_JOB_ID') is not None
    walltime: str = '06:00:00' if is_slurm is not None else None


@dataclass
class Config:

    gpt: GPTConfig = GPTConfig()
    io_metrics: IOMetricsConfig = IOMetricsConfig()
    data: DataConfig = DataConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    learning_rate: LearningRateConfig = LearningRateConfig()
    ddp: DDPConfig = DDPConfig()
    system: SystemConfig = SystemConfig()
    sampling: SamplingConfig = SamplingConfig()

    
    
    
    
