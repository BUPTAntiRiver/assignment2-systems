"""
Basic end to end benchmarking script of forward and backward passes.
The script does the following:
    1. Creates a model with given hyperparameters.
    2. Generates random input data and target labels.
    3. Run warmup then measure
"""

import cs336_basics.model
import cs336_basics.nn_utils
import torch
import torch.cuda.nvtx as nvtx
import timeit
import argparse
import math
from tqdm import tqdm


parser = argparse.ArgumentParser()

# Model hyperparameters
parser.add_argument("--d_model", type=int, default=128, help="Dimension of model")
parser.add_argument("--d_ff", type=int, default=512, help="Dimension of feedforward layer")
parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
parser.add_argument("--context_length", type=int, default=128, help="Context length")
parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")

# Benchmarking hyperparameters
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--warmup_iters", type=int, default=5, help="Number of warmup iterations")
parser.add_argument("--measure_iters", type=int, default=10, help="Number of measurement iterations")
parser.add_argument("--dtype", type=str, default="float32", help="Data type to use (float32, float16, bfloat16)")

parser.add_argument(
    "--include_loss_in_forward",
    action="store_true",
    help="If set, forward timing includes loss computation.",
)
parser.add_argument(
    "--optimizer_step",
    action="store_true",
    help="If set, runs optimizer.step() and times it separately.",
)
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW (if --optimizer_step)")
parser.add_argument(
    "--use_tqdm",
    action="store_true",
    help="If set, show progress bars (can add overhead for small benchmarks).",
)
parser.add_argument("--compile", action="store_true", help="If set, use torch.compile on the model.")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model
model = cs336_basics.model.BasicsTransformerLM(
    vocab_size=args.vocab_size,
    context_length=args.context_length,
    d_model=args.d_model,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    d_ff=args.d_ff,
    rope_theta=100000,
)

if args.compile:
    model = torch.compile(model)

model.to(device)
model.train()

optimizer = None
if args.optimizer_step:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

# Generate random input data and target labels
input_data = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length)).to(device)
target_labels = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length)).to(device)


def _sync():
    if device.type == "cuda":
        torch.cuda.synchronize()


def _summarize(times_s):
    if not times_s:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0}
    mean = sum(times_s) / len(times_s)
    if len(times_s) > 1:
        var = sum((x - mean) ** 2 for x in times_s) / (len(times_s) - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    sorted_times = sorted(times_s)
    p50 = sorted_times[len(sorted_times) // 2]
    p95_idx = max(0, math.ceil(0.95 * len(sorted_times)) - 1)
    p95 = sorted_times[p95_idx]
    return {"mean": mean, "std": std, "p50": p50, "p95": p95}

@nvtx.range("benchmark_iteration")
def benchmark_iteration():
    # Important for correctness (and stable perf): don't accumulate grads across iters.
    model.zero_grad(set_to_none=True)

    # Ensure prior async GPU work is not attributed to this iter.
    _sync()
    with nvtx.range("forward"):
        forward_start = timeit.default_timer()

        with torch.autocast(device_type=device.type, dtype=getattr(torch, args.dtype), cache_enabled=True):
            logits = model(input_data)
            loss = None
            if args.include_loss_in_forward:
                loss = cs336_basics.nn_utils.cross_entropy(
                    logits.reshape(-1, args.vocab_size), target_labels.reshape(-1)
                )

    _sync()
    forward_end = timeit.default_timer()

    if loss is None:
        loss = cs336_basics.nn_utils.cross_entropy(
            logits.reshape(-1, args.vocab_size), target_labels.reshape(-1)
        )

    _sync()
    with nvtx.range("backward"):
        backward_start = timeit.default_timer()
        loss.backward()
        _sync()
        backward_end = timeit.default_timer()

    opt_time = 0.0
    if optimizer is not None:
        _sync()
        with nvtx.range("optimizer_step"):
            opt_start = timeit.default_timer()
            optimizer.step()
        _sync()
        opt_end = timeit.default_timer()
        opt_time = opt_end - opt_start

    return (forward_end - forward_start), (backward_end - backward_start), opt_time

def main():
    # Warmup
    warmup_iter = range(args.warmup_iters)
    if args.use_tqdm:
        warmup_iter = tqdm(warmup_iter, desc="Warmup")
    for _ in warmup_iter:
        benchmark_iteration()

    # Measurement
    f_times = []
    b_times = []
    o_times = []
    measure_iter = range(args.measure_iters)
    if args.use_tqdm:
        measure_iter = tqdm(measure_iter, desc="Measurement")

    if device.type == "cuda":
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)
    for _ in measure_iter:
        f_time, b_time, o_time = benchmark_iteration()
        f_times.append(f_time)
        b_times.append(b_time)
        if optimizer is not None:
            o_times.append(o_time)
    if device.type == "cuda":
        torch.cuda.memory._dump_memory_history("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
    f = _summarize(f_times)
    b = _summarize(b_times)
    o = _summarize(o_times) if optimizer is not None else None

    tokens_per_iter = args.batch_size * args.context_length
    step_mean = f["mean"] + b["mean"] + (o["mean"] if o is not None else 0.0)

    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}, Context length: {args.context_length} (tokens/iter={tokens_per_iter})")
    print(f"Warmup iters: {args.warmup_iters}, Measure iters: {args.measure_iters}")
    print(f"Include loss in forward: {args.include_loss_in_forward}")
    print(f"Optimizer step: {optimizer is not None}")
    print()

    print(f"Forward (s): mean={f['mean']:.6f} std={f['std']:.6f} p50={f['p50']:.6f} p95={f['p95']:.6f}")
    print(f"Backward (s): mean={b['mean']:.6f} std={b['std']:.6f} p50={b['p50']:.6f} p95={b['p95']:.6f}")
    if o is not None:
        print(f"Opt step (s): mean={o['mean']:.6f} std={o['std']:.6f} p50={o['p50']:.6f} p95={o['p95']:.6f}")
    print(f"Step total (s): mean={step_mean:.6f} (throughput={tokens_per_iter / step_mean:.2f} tokens/s)")

if __name__ == "__main__":
    main()
