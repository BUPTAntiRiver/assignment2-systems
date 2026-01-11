import torch
import torch.distributed as dist


class ShardedOptimizerWrapper(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.param_to_rank = {}
        self.params_list = list(params)
        super().__init__(self.params_list, kwargs)
        local_params = self.params_list[self.rank::self.world_size]
        self.optimizer = optimizer_cls(local_params, **kwargs)


    def step(self, closure=None, **kwargs):
        # perform local step
        self.optimizer.step(closure, **kwargs)
    
        # broadcast updated params
        for param in self.param_to_rank:
            dist.broadcast(param.data, src=self.param_to_rank[param])


    def add_param_group(self, param_group):
        self.param_groups.append(param_group)
        # update param_to_rank mapping
        for r in range(self.world_size):
            for p in param_group['params'][r::self.world_size]:
                self.param_to_rank[p] = r
