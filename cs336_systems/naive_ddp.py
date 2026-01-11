import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time
from cs336_basics.model import BasicsTransformerLM


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def distributed_train(rank, world_size):
    setup(rank, world_size)
    
    model = BasicsTransformerLM(
        vocab_size=64,
        context_length=64,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        rope_theta=100000,
    )
    
    # Broadcast each parameter from rank 0 to all ranks
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    local_bs = 8  # Each rank processes 8 examples
    total_times = []
    comm_times = []
    
    for step in range(5):
        iter_start = time.time()
        
        optimizer.zero_grad()
        
        # Generate all data on rank 0, broadcast to all ranks
        if rank == 0:
            torch.manual_seed(42 + step)
            all_input_ids = torch.randint(0, 64, (world_size * local_bs, 64))
            all_target_ids = torch.randint(0, 64, (world_size * local_bs, 64))
        else:
            all_input_ids = torch.empty((world_size * local_bs, 64), dtype=torch.long)
            all_target_ids = torch.empty((world_size * local_bs, 64), dtype=torch.long)
        
        dist.broadcast(all_input_ids, src=0)
        dist.broadcast(all_target_ids, src=0)
        
        # Each rank processes a disjoint subset
        offset = rank * local_bs
        input_ids = all_input_ids[offset : offset + local_bs]
        target_ids = all_target_ids[offset : offset + local_bs]
        
        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, 64), target_ids.view(-1))
        loss.backward()
        
        # Measure communication time
        comm_start = time.time()
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size
        comm_end = time.time()
        comm_times.append(comm_end - comm_start)
        
        optimizer.step()
        
        iter_end = time.time()
        total_times.append(iter_end - iter_start)
    
    # Print timing results from rank 0
    if rank == 0:
        avg_total = sum(total_times) / len(total_times)
        avg_comm = sum(comm_times) / len(comm_times)
        comm_ratio = (avg_comm / avg_total) * 100
        print(f"Average total time per iteration: {avg_total:.6f} seconds")
        print(f"Average communication time per iteration: {avg_comm:.6f} seconds")
        print(f"Communication time proportion: {comm_ratio:.2f}%")
    
    dist.destroy_process_group()


def train_distributed(world_size=4):
    mp.spawn(fn=distributed_train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    train_distributed(world_size=2)