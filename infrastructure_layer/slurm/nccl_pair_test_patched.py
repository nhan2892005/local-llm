import datetime
import os
import socket
import time

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])

print(
    f"START host={socket.getfqdn()} rank={rank} "
    f"MASTER={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']} "
    f"NCCL_NET={os.getenv('NCCL_NET')} "
    f"NCCL_PORT_MIN={os.getenv('NCCL_PORT_MIN')} "
    f"NCCL_PORT_MAX={os.getenv('NCCL_PORT_MAX')}",
    flush=True,
)

print(
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"nccl={torch.cuda.nccl.version()}",
    flush=True,
)

torch.cuda.set_device(0)

dist.init_process_group(
    backend="nccl",
    init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
    rank=rank,
    world_size=2,
    timeout=datetime.timedelta(seconds=90),
)

print("NCCL_INIT_OK", flush=True)

x = torch.tensor([rank + 1], dtype=torch.int64, device="cuda")
dist.all_reduce(x)
torch.cuda.synchronize()

value = int(x.item())
print(f"NCCL_OK rank={rank} all_reduce={value}", flush=True)

if value != 3:
    raise RuntimeError(f"unexpected all_reduce result {value}")

dist.barrier(device_ids=[0])
time.sleep(2)
dist.destroy_process_group()
print(f"DONE rank={rank}", flush=True)
