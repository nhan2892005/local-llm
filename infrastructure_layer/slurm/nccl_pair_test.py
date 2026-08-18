import datetime
import os
import socket
import sys
import time

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world = 2

print(
    f"START host={socket.getfqdn()} rank={rank} "
    f"MASTER={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']} "
    f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')} "
    f"NCCL_NET={os.getenv('NCCL_NET')} "
    f"NCCL_IB_DISABLE={os.getenv('NCCL_IB_DISABLE')} "
    f"NCCL_SOCKET_IFNAME={os.getenv('NCCL_SOCKET_IFNAME')}",
    flush=True,
)

print(
    f"TORCH torch={torch.__version__} "
    f"cuda_build={torch.version.cuda} "
    f"cuda_available={torch.cuda.is_available()} "
    f"device_count={torch.cuda.device_count()} "
    f"nccl_version={torch.cuda.nccl.version()}",
    flush=True,
)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")
if torch.cuda.device_count() < 1:
    raise RuntimeError("No CUDA device visible")

torch.cuda.set_device(0)

print("NCCL_INIT_BEGIN", flush=True)

dist.init_process_group(
    backend="nccl",
    init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
    rank=rank,
    world_size=world,
    timeout=datetime.timedelta(seconds=90),
)

print("NCCL_INIT_OK", flush=True)

# Keep the communicator alive long enough for the shell-side socket monitor.
time.sleep(3)

x = torch.tensor([rank + 1], dtype=torch.int64, device="cuda")
dist.all_reduce(x)
torch.cuda.synchronize()

value = int(x.item())
print(
    f"NCCL_OK host={socket.getfqdn()} rank={rank} all_reduce={value}",
    flush=True,
)

if value != 3:
    raise RuntimeError(f"unexpected all_reduce result: {value}")

dist.barrier(device_ids=[0])
time.sleep(2)
dist.destroy_process_group()

print(f"DONE rank={rank}", flush=True)
