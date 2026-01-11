import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit
import pandas as pd

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_all_reduce(rank, world_size, data_size=1024**2):
    setup(rank, world_size)
    data = torch.randn((data_size // 4, ), dtype=torch.float32)
    #print(f"rank {rank} data (before all-reduce): {data}")
    dist.all_reduce(data, async_op=False)
    #print(f"rank {rank} data (after all-reduce): {data}")
    dist.destroy_process_group()

def all_reduce(data_size=1024**2, world_size=4):
    mp.spawn(fn=distributed_all_reduce, args=(world_size, data_size), nprocs=world_size, join=True)

if __name__ == "__main__":
    # warm-up run
    print("Warming up...")
    for _ in range(5):
        all_reduce()

    results = []
    # measure
    for data_size in [1024**2, 10 * 1024**2, 100 * 1024**2, 1024 ** 3]:
        for world_size in [2, 4, 6]:
            print(f"\nMeasuring all-reduce with data size: {data_size / (1024**2)} MB and world size: {world_size}")
            num_runs = 10
            elapsed_time = timeit.timeit(lambda: all_reduce(data_size, world_size), number=num_runs)
            avg_time_per_run = elapsed_time / num_runs
            print(f"Average time per run over {num_runs} runs: {avg_time_per_run:.6f} seconds")
            results.append({
                "data_size_MB": data_size / (1024**2),
                "world_size": world_size,
                "avg_time_per_run_sec": avg_time_per_run
            })

    df = pd.DataFrame(results)
    print("\nFinal Results:")
    print(df.to_markdown(index=False))
