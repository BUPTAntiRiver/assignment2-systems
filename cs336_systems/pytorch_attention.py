from cs336_basics.model import scaled_dot_product_attention
import torch
import timeit
import torch.cuda.nvtx as nvtx
import pandas as pd


batch_size = 8 # not multihead
d_models = [16, 32, 64, 128]
seq_lens = [256, 1024, 4096, 8192, 16384]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sync = torch.cuda.synchronize if device == 'cuda' else lambda: None
results = []

kernel = torch.compile(scaled_dot_product_attention)

for d_model in d_models:
    for seq_len in seq_lens:
        q = torch.randn((batch_size, seq_len, d_model), device=device, requires_grad=True)
        k = torch.randn((batch_size, seq_len, d_model), device=device, requires_grad=True)
        v = torch.randn((batch_size, seq_len, d_model), device=device, requires_grad=True)

        grad_output = torch.ones_like(q)

        # Warmup
        for _ in range(5):
            out = kernel(q, k, v)
            out.backward(grad_output)
            q.grad.zero_()
            k.grad.zero_()
            v.grad.zero_()

        # Measurement
        # 1. Forward time
        sync()
        start = timeit.default_timer()
        with torch.no_grad():
            for _ in range(100):
                _ = kernel(q, k, v)
        sync()
        end = timeit.default_timer()
        forward_time = (end - start) / 100.0

        # 2. Measure memory usage
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            out = kernel(q, k, v)
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # in MB
        else:
            peak_memory = 0.0  # Not applicable for CPU

        # 3. Backward time
        
        backward_time = 0.0
        for _ in range(100):
            out = kernel(q, k, v)
            sync()
            start = timeit.default_timer()
            out.backward(grad_output)
            sync()
            end = timeit.default_timer()
            backward_time += (end - start) 
            q.grad = k.grad = v.grad = None
        backward_time /= 100.0

        print(f"d_model: {d_model}, seq_len: {seq_len}, forward_time: {forward_time:.6f}s, backward_time: {backward_time:.6f}s, peak_memory: {peak_memory:.2f}MB")

        results.append({
            'd_model': d_model,
            'seq_len': seq_len,
            'forward_time_s': forward_time,
            'backward_time_s': backward_time,
            'peak_memory_MB': peak_memory
        })

table = pd.DataFrame(results)
print(table)
