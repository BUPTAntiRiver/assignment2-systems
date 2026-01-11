import os
import pathlib
import tracemalloc
import torch
import torch.multiprocessing as mp
from copy import deepcopy
from torch import nn
from tests.common import ToyModel, ToyModelWithTiedWeights, _setup_process_group, _cleanup_process_group
from tests.adapters import get_sharded_optimizer


class MemoryProfiler:
    def __init__(self):
        self.snapshots = []

    def snapshot(self, label):
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append((label, snapshot))

    def compare(self, label1, label2):
        snapshot1 = next(s for l, s in self.snapshots if l == label1)
        snapshot2 = next(s for l, s in self.snapshots if l == label2)
        stats = snapshot2.compare_to(snapshot1, 'lineno')
        print(f"\n=== Memory difference: {label1} -> {label2} ===")
        for stat in stats[:10]:
            print(stat)
        total_diff = sum(stat.size_diff for stat in stats) / 1024 / 1024
        print(f"Total memory change: {total_diff:.2f} MB")

    def print_summary(self):
        print("\n=== Memory Snapshot Summary ===")
        for label, snapshot in self.snapshots:
            stats = snapshot.statistics('lineno')
            total = sum(stat.size for stat in stats) / 1024 / 1024
            print(f"{label}: {total:.2f} MB")


def profile_training(rank, world_size, model_class, use_sharded=True):
    tracemalloc.start()
    profiler = MemoryProfiler()
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    torch.manual_seed(42)

    profiler.snapshot("init")

    model = model_class().to(device)
    profiler.snapshot("model_created")

    optimizer_cls = torch.optim.AdamW
    if use_sharded:
        optimizer = get_sharded_optimizer(
            model.parameters(),
            optimizer_cls,
            lr=0.1,
            weight_decay=0.1,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    else:
        optimizer = optimizer_cls(
            model.parameters(),
            lr=0.1,
            weight_decay=0.1,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    profiler.snapshot("optimizer_created")

    for step in range(10):
        optimizer.zero_grad()

        input_ = torch.rand((32, 10)).to(device)
        labels = torch.rand((32, 5)).to(device)

        logits = model(input_)
        loss = ((labels - logits) ** 2).sum()
        loss.backward()

        if step == 0:
            profiler.snapshot("first_backward")

        optimizer.step()

        if step == 0:
            profiler.snapshot("first_step")
        if step == 4:
            profiler.snapshot("after_5_steps")

    profiler.snapshot("after_10_steps")

    if rank == 0:
        profiler.print_summary()
        profiler.compare("init", "model_created")
        profiler.compare("model_created", "optimizer_created")
        profiler.compare("optimizer_created", "first_backward")
        profiler.compare("first_backward", "first_step")
        profiler.compare("first_step", "after_5_steps")
        profiler.compare("after_5_steps", "after_10_steps")

    _cleanup_process_group()


def main():
    print("Starting memory profiling...")
    tracemalloc.start()

    world_size = 2

    print("\n" + "="*60)
    print("Profiling NON-SHARDED optimizer")
    print("="*60)
    mp.spawn(
        profile_training,
        args=(world_size, ToyModel, False),
        nprocs=world_size,
        join=True,
    )

    tracemalloc.stop()
    tracemalloc.clear_traces()
    tracemalloc.start()

    print("\n" + "="*60)
    print("Profiling SHARDED optimizer")
    print("="*60)
    mp.spawn(
        profile_training,
        args=(world_size, ToyModel, True),
        nprocs=world_size,
        join=True,
    )

    tracemalloc.stop()


if __name__ == "__main__":
    main()