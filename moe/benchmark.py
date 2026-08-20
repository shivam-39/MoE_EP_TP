import argparse
import time

import numpy as np
from rng import get_rng, register_rng
from mpiwrapper import mpi
from moe import SimpleMoE, MoE_EP, MoE_TP


def run_moe(
    moe_type="tp",
    batch_size=8,
    feature_dim=32,
    hidden_dim=128,
    output_dim=64,
    num_experts=None,
    topk=2,
    warmup=2,
    repeats=5,
):
    """
    Unified function to run different types of MoE models

    Args:
        moe_type: Type of MoE to run ("simple", "ep", or "tp")
        batch_size: Number of samples in the batch
        feature_dim: Dimension of input features
        hidden_dim: Hidden dimension for experts
        output_dim: Output dimension
        topk: Number of experts to route each input to
    """
    # Get number of experts based on MPI world size
    num_experts = mpi.get_size()

    # Generate input data
    np.random.seed(0)
    X = np.random.randn(batch_size, feature_dim)

    if moe_type != "simple":
        # Synchronize the input data across all processes
        if mpi.get_rank() == 0:
            X = get_rng().randn(batch_size, feature_dim)
        else:
            X = None
        X = mpi.comm.bcast(X, root=0)

    # Create appropriate MoE model
    model_class = {
        "simple": SimpleMoE,
        "ep": MoE_EP,
        "tp": MoE_TP,
    }.get(moe_type, MoE_TP)

    moe = model_class(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        topk=topk,
    )

    # Warm up (and make sure all ranks finish before timing)
    for _ in range(warmup):
        _ = moe(X)
    mpi.barrier()

    start_time = time.perf_counter()
    for _ in range(repeats):
        outputs = moe(X)
    mpi.barrier()
    avg_duration_ms = 1000 * (time.perf_counter() - start_time) / repeats

    if mpi.get_rank() == 0:
        print(
            f"{moe_type:6s}  batch={batch_size:4d}  hidden={hidden_dim:4d}  "
            f"out={output_dim:4d}  topk={topk}  world={mpi.get_size()}  "
            f"{avg_duration_ms:8.3f} ms"
        )

    return dict(outputs=outputs, avg_duration_ms=avg_duration_ms)


def _can_run_tp(hidden_dim, output_dim):
    world = mpi.get_size()
    return hidden_dim % world == 0 and output_dim % world == 0


def benchmark_moe(repeats=5):
    """Sweep batch size, expert width, and top-k for Simple / TP / EP MoE."""
    rank = mpi.get_rank()
    register_rng("expert_with_rank", np.random.RandomState(rank + 100))

    configs = []
    # Scale batch (activation volume vs routing overhead)
    for batch_size in (16, 64, 256):
        configs.append(dict(batch_size=batch_size, feature_dim=32, hidden_dim=128, output_dim=64, topk=2))
    # Scale expert hidden width (TP shards this dimension)
    for hidden_dim in (64, 128, 256):
        configs.append(dict(batch_size=64, feature_dim=32, hidden_dim=hidden_dim, output_dim=64, topk=2))
    # Scale top-k (how many experts fire per token)
    for topk in (1, 2, 4):
        configs.append(dict(batch_size=64, feature_dim=32, hidden_dim=128, output_dim=64, topk=topk))

    results = []
    for cfg in configs:
        row = dict(cfg)
        row["world_size"] = mpi.get_size()
        for moe_type in ("simple", "tp", "ep"):
            if moe_type == "tp" and not _can_run_tp(cfg["hidden_dim"], cfg["output_dim"]):
                row[moe_type] = None
                continue
            result = run_moe(moe_type=moe_type, repeats=repeats, **cfg)
            row[moe_type] = result["avg_duration_ms"]
        results.append(row)

    if rank == 0:
        print("\n=== Summary (ms / forward) ===")
        print(
            f"{'batch':>6} {'hidden':>6} {'out':>5} {'topk':>5} {'world':>5} "
            f"{'simple':>10} {'tp':>10} {'ep':>10} {'tp/ep':>8}"
        )
        for row in results:
            tp = row["tp"]
            ep = row["ep"]
            ratio = f"{tp / ep:8.2f}" if tp and ep else f"{'n/a':>8}"
            tp_s = f"{tp:10.3f}" if tp is not None else f"{'n/a':>10}"
            print(
                f"{row['batch_size']:6d} {row['hidden_dim']:6d} {row['output_dim']:5d} "
                f"{row['topk']:5d} {row['world_size']:5d} "
                f"{row['simple']:10.3f} {tp_s} {ep:10.3f} {ratio}"
            )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Simple / TP / EP MoE")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    benchmark_moe(repeats=args.repeats)
