import os
import subprocess

import torch
import torch.distributed as dist


def setup_distributed(backend="nccl"):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()
    
    rank = int(os.environ["RANK"])
    
    world_size = int(os.environ["WORLD_SIZE"])
    
    # rank = 0
    # world_size = 1

    torch.cuda.set_device(rank % num_gpus)

########################################
    
    # os.environ["MASTER_ADDR"] = 'localhost'
    # os.environ["MASTER_PORT"] = '5678'
    ##########################################
    
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )

    return rank, world_size
