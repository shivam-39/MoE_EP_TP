# TP vs EP MoE Performance

Benchmarks ran on a 10-core CPU with `mpirun -n 4` (one expert per rank) and `OMP_NUM_THREADS=1` so the four ranks do not oversubscribe BLAS. Times are milliseconds per forward. Collectives go through the assignment wrapper (`allreduce` / `alltoall` of NumPy objects).

## Results (world size = 4)

| batch | hidden | out | topk | Simple | TP | EP | TP / EP |
|------:|-------:|----:|-----:|-------:|---:|---:|--------:|
| 16 | 128 | 64 | 2 | 0.87 | 0.55 | **0.26** | 2.13 |
| 64 | 128 | 64 | 2 | 1.28 | 0.96 | **0.55** | 1.75 |
| 256 | 128 | 64 | 2 | 4.84 | 2.91 | **1.87** | 1.56 |
| 64 | 64 | 64 | 2 | 1.14 | 1.00 | **0.56** | 1.77 |
| 64 | 256 | 64 | 2 | 1.89 | 1.32 | **0.83** | 1.60 |
| 64 | 128 | 64 | 1 | 0.67 | 0.87 | **0.38** | 2.28 |
| 64 | 128 | 64 | 4 | 3.19 | 1.55 | **0.98** | 1.58 |

Reproduce:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mpirun -n 4 python benchmark.py --repeats 3
```

TP requires `hidden_dim` and `output_dim` to be divisible by the world size.

## What each implementation communicates

**Tensor parallel (`MoE_TP`).** Every rank holds a column-shard of *every* expert. A `ShardedLinear` computes `x W_local + b_local`, writes that slice into a full-width buffer, and **all-reduces** so all ranks reconstruct the same activation. The router is replicated, so ranks stay in lockstep. Each GEMM is `1/P` of the full matmul, but **every rank still walks every active expert on the full batch**, and each linear layer pays a dense all-reduce of size `O(batch × out_features)`.

**Expert parallel (`MoE_EP`).** Rank `i` owns expert `i` in full. Tokens are **dispatched with all-to-all**, the local expert runs only on tokens that routed here, and a reverse all-to-all returns the outputs for a gated sum. Compute is partitioned by expert. Communication is two irregular all-to-alls whose volume is `O(batch × topk × dim)` rather than `O(num_active_experts × batch × dim)`.

## Why EP is faster here

With one BLAS thread per rank, **EP is 1.6–2.3× faster than TP** and both usually beat `SimpleMoE`.

The compute split is the main reason. For `topk=2` and 4 experts, each token uses two experts, so EP evaluates about `batch × topk / P` tokens per rank through **one full expert**. TP instead evaluates a `1/P` shard of **every active expert on all `batch` tokens**. When routing hits all four experts, that is roughly twice the GEMM work per rank, plus an all-reduce after every sharded linear.

That also explains the sweeps:

- **Batch.** EP stays faster from 16 to 256. The TP/EP ratio falls slightly (2.13 → 1.56) as GEMMs get larger and collective overhead is a smaller fraction of runtime.
- **Hidden width.** Widening the expert (64 → 256) helps both; EP remains ahead because it still avoids replicating expert compute.
- **top-k.** `topk=1` is EP’s best case (0.38 ms vs TP 0.87 ms); TP even loses to SimpleMoE because all-reduce costs more than the small sharded GEMM saves. `topk=4` forces every expert to run on every token; EP still wins (0.98 vs 1.55) because each rank runs a single expert and two all-to-alls, while TP all-reduces four experts × two layers.

## When TP would win

TP is the better tool when the *expert itself* is the thing to shard: a very wide GEMM, a small expert count, or a batch that already fills the device so EP would be load-imbalanced. A production TP stack would also fuse column-parallel `fc1` with row-parallel `fc2` to drop an all-reduce; this assignment shards *output* on both layers, so TP pays two all-reduces per expert. EP would also look worse if all-to-all were the bottleneck (tiny messages, poorly balanced routing). In this NumPy/MPI microbenchmark those effects do not outweigh EP’s extra compute parallelism.

**Takeaway:** for this MoE (few experts, modest width), expert parallelism is the right default — each rank does one expert’s work on a subset of tokens. Tensor parallelism is the right default when you need to split a single large GEMM, not when you already have several independent experts to spread across ranks.
