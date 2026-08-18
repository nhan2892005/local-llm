import os
import socket
import sys
import time

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world = 2

def listen_ports():
    ports = set()
    for fn in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(fn) as f:
                next(f)
                for line in f:
                    fields = line.split()
                    local = fields[1]
                    state = fields[3]
                    if state != "0A":  # LISTEN
                        continue
                    port = int(local.split(":")[1], 16)
                    ports.add(port)
        except FileNotFoundError:
            pass
    return sorted(ports)

print(
    f"START host={socket.getfqdn()} rank={rank} "
    f"MASTER={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']} "
    f"IF={os.getenv('GLOO_SOCKET_IFNAME')} "
    f"MIN={os.getenv('GLOO_PORT_MIN')} MAX={os.getenv('GLOO_PORT_MAX')}",
    flush=True,
)

dist.init_process_group(
    backend="gloo",
    init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
    rank=rank,
    world_size=world,
)

ports = listen_ports()
gloo_min = int(os.environ["GLOO_PORT_MIN"])
gloo_max = int(os.environ["GLOO_PORT_MAX"])
range_ports = [p for p in ports if gloo_min <= p <= gloo_max]

print(f"LISTEN_PORTS rank={rank} all={ports}", flush=True)
print(f"GLOO_RANGE_LISTENERS rank={rank} ports={range_ports}", flush=True)

if not range_ports:
    raise RuntimeError(
        f"No LISTEN socket observed inside patched Gloo range "
        f"{gloo_min}-{gloo_max}"
    )

x = torch.tensor([rank + 1], dtype=torch.int64)
dist.all_reduce(x)
print(
    f"GLOO_OK host={socket.getfqdn()} rank={rank} all_reduce={x.item()}",
    flush=True,
)

if x.item() != 3:
    raise RuntimeError(f"unexpected all_reduce result: {x.item()}")

dist.barrier()
time.sleep(2)
dist.destroy_process_group()
print(f"DONE rank={rank}", flush=True)
