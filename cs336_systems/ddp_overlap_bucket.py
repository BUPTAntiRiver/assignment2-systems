import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time
from cs336_basics.model import BasicsTransformerLM


class DDP_bucket(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = model
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        self.current_bucket = []
        self.current_bucket_size = 0

        self.param_to_bucket_id = {}
        self.buckets = []
        self.bucket_ready_count = []

        current_bucket = []
        current_size = 0

        for param in reversed(list(self.module.parameters())):
            if not param.requires_grad:
                continue

            current_bucket.append(param)
            current_size += param.numel() * param.element_size()

            self.param_to_bucket_id[param] = len(self.buckets)

            if current_size >= self.bucket_size_bytes:
                self.buckets.append(current_bucket)
                self.bucket_ready_count.append(0)
                current_bucket = []
                current_size = 0
        
        if current_bucket:
            self.buckets.append(current_bucket)
            self.bucket_ready_count.append(0)

        # Broadcast parameters from rank 0 to all other ranks
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # Register hooks for gradient synchronization (only on params that require grad)
        self.params_to_hook = [p for p in self.module.parameters() if p.requires_grad]
        for param in self.params_to_hook:
            param.register_post_accumulate_grad_hook(self.grad_hook)

        self.work_handles = [None] * len(self.buckets)

    def grad_hook(self, param):
        bucket_id = self.param_to_bucket_id[param]
        self.bucket_ready_count[bucket_id] += 1
        
        if self.bucket_ready_count[bucket_id] == len(self.buckets[bucket_id]):
            self._all_reduce_current_bucket(bucket_id)

    def _all_reduce_current_bucket(self, bucket_id):
        params = self.buckets[bucket_id]
        dense_grads = [p.grad for p in params]
        flat_grad = torch._utils._flatten_dense_tensors(dense_grads)
        work = dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, async_op=True)
        self.work_handles[bucket_id] = (work, dense_grads, flat_grad)

    def forward(self, *inputs, **kwargs):
        self.bucket_ready_count = [0 for _ in self.buckets]
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()
        for i in range(len(self.work_handles)):
            if self.work_handles[i] is None:
                continue
            work, dense_grads, flat_grad = self.work_handles[i]
            work.wait()
            flat_grad.div_(world_size)
            grads_coalesced = torch._utils._unflatten_dense_tensors(flat_grad, dense_grads)
            for p_grad, coalesced in zip(dense_grads, grads_coalesced):
                p_grad.copy_(coalesced)
            self.work_handles[i] = None


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29502"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def distributed_train(rank, world_size, bucket_size_mb):
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
    ddp_model = DDP_bucket(model, bucket_size_mb=bucket_size_mb)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)
    ddp_model.train()
    
    local_bs = 16  # Each rank processes 16 examples
    total_times = []

    # warm-up iterations
    for _ in range(5):
        optimizer.zero_grad()
        
        # Generate all data on rank 0, broadcast to all ranks
        input_ids = torch.randint(0, 512, (local_bs, 128), dtype=torch.long)
        target_ids = torch.randint(0, 512, (local_bs, 128), dtype=torch.long)
        
        outputs = ddp_model(input_ids)
        loss = criterion(outputs.view(-1, 512), target_ids.view(-1))
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()
    
    for _ in range(10):
        optimizer.zero_grad()
        
        # Generate all data on rank 0, broadcast to all ranks
        input_ids = torch.randint(0, 512, (local_bs, 128), dtype=torch.long)
        target_ids = torch.randint(0, 512, (local_bs, 128), dtype=torch.long)
        
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


def train_distributed(world_size=4, bucket_size_mb=1.0):
    mp.spawn(fn=distributed_train, args=(world_size, bucket_size_mb), nprocs=world_size, join=True)


if __name__ == "__main__":
    for bucket_size_mb in [0.01, 1, 10, 100, 1000]:
        print(f"\nTraining with bucket size: {bucket_size_mb} MB")
        train_distributed(world_size=2, bucket_size_mb=bucket_size_mb)