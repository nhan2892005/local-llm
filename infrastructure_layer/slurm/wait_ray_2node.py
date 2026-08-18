import os
import sys
import time
import ray

timeout = int(os.environ.get("RAY_WAIT_TIMEOUT", "1200"))
deadline = time.time() + timeout

ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")

last = None
while time.time() < deadline:
    alive = [n for n in ray.nodes() if n.get("Alive")]
    resources = ray.cluster_resources()
    gpus = float(resources.get("GPU", 0))
    cpus = float(resources.get("CPU", 0))
    state = (len(alive), gpus, cpus)

    if state != last:
        print(f"RAY_WAIT nodes={len(alive)} gpus={gpus:g} cpus={cpus:g}", flush=True)
        for n in alive:
            print(
                "  node="
                + str(n.get("NodeManagerAddress"))
                + " alive="
                + str(n.get("Alive")),
                flush=True,
            )
        last = state

    if len(alive) >= 2 and gpus >= 8:
        print("RAY_CLUSTER_READY nodes>=2 gpus>=8", flush=True)
        ray.shutdown()
        sys.exit(0)

    time.sleep(3)

print("ERROR: timed out waiting for 2 Ray nodes / 8 GPUs", flush=True)
ray.shutdown()
sys.exit(2)
