import torch
import torch.distributed as dist

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
