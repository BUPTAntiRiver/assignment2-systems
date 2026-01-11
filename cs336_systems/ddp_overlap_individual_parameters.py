import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time
from cs336_basics.model import BasicsTransformerLM

class DDP(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.module = model
        self.work_handles = []

        # Broadcast parameters from rank 0 to all other ranks
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # Register hooks for gradient synchronization (only on params that require grad)
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self.grad_hook)

    def grad_hook(self, param):
        if param.grad is None:
            return
        work = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self.work_handles.append(work)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for work in self.work_handles:
            work.wait()

        self.work_handles.clear()

        world_size = dist.get_world_size()
        for param in self.module.parameters():
            if param.requires_grad:
                param.grad.detach().div_(world_size)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29502"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def distributed_train(rank, world_size):
    setup(rank, world_size)
    
    model = BasicsTransformerLM(
        vocab_size=512,
        context_length=128,
        d_model=256,
        num_layers=4,
        num_heads=8,
        d_ff=1024,
        rope_theta=100000,
    )
    
    # Wrap with DDP
    ddp_model = DDP(model)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)
    ddp_model.train()
    
    local_bs = 16  # Each rank processes 16 examples
    total_times = []
    
    for step in range(5):
        optimizer.zero_grad()
        
        # Generate all data on rank 0, broadcast to all ranks
        if rank == 0:
            torch.manual_seed(42 + step)
            all_input_ids = torch.randint(0, 512, (world_size * local_bs, 128))
            all_target_ids = torch.randint(0, 512, (world_size * local_bs, 128))
        else:
            all_input_ids = torch.empty((world_size * local_bs, 128), dtype=torch.long)
            all_target_ids = torch.empty((world_size * local_bs, 128), dtype=torch.long)
        
        dist.broadcast(all_input_ids, src=0)
        dist.broadcast(all_target_ids, src=0)
        
        # Each rank processes a disjoint subset
        offset = rank * local_bs
        input_ids = all_input_ids[offset : offset + local_bs]
        target_ids = all_target_ids[offset : offset + local_bs]
        
        iter_start = time.time()
        
        outputs = ddp_model(input_ids)
        loss = criterion(outputs.view(-1, 512), target_ids.view(-1))
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()
        
        iter_end = time.time()
        total_times.append(iter_end - iter_start)
    
    # Print timing results from rank 0
    if rank == 0:
        avg_total = sum(total_times) / len(total_times)
        print(f"DDP Overlap - Average total time per iteration: {avg_total:.6f} seconds")
    
    dist.destroy_process_group()


def train_distributed(world_size=4):
    mp.spawn(fn=distributed_train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    train_distributed(world_size=2)
