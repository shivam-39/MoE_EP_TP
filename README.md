# Mixture of Experts: Expert Parallel and Tensor Parallel

Distributed MoE in NumPy + MPI, with two communication patterns for the expert layer:

- **Tensor parallel (TP):** every rank holds a column-shard of *every* expert. Partial GEMMs are combined with all-reduce.
- **Expert parallel (EP):** each rank owns one full expert. Tokens are dispatched and results returned with all-to-all.

The repo also includes the supporting pieces used to build that MoE: manual MPI collectives, data/model-parallel transformer communication, and a fused Triton `matmul + add + ReLU` kernel.

## Layout

```text
moe/                 MoE models, tests, and TP vs EP benchmark
  moe.py             SimpleMoE, ShardedLinear, MoE_TP, MoE_EP
  test_moe.py
  benchmark.py
  analysis.md
parallel/            MPI collectives and transformer parallel comms
  mpi_wrapper/       Allreduce / Alltoall (library + from-scratch)
  data/              data-parallel input split
  model/             get_info + naive model-parallel forward/backward
  tests/
  mpi-test.py
kernels/
  matmul_triton.ipynb   fused D = ReLU(A @ B + C) in fp16
```

## Setup

Needs a machine with MPI and at least a few CPU cores (8 is enough for the parallel tests).

```bash
# macOS
brew install openmpi

# Linux
sudo apt install libopenmpi-dev
```

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "setuptools<72" wheel   # mpi4py 3.1.5 needs an older setuptools
pip install mpi4py==3.1.5 --no-build-isolation
pip install -r requirements.txt
```

## Mixture of Experts

`moe/moe.py` implements three variants:

| Class | Placement | Communication |
|---|---|---|
| `SimpleMoE` | all experts on one process | none (reference) |
| `MoE_TP` | every expert, sharded on the output dim | all-reduce after each `ShardedLinear` |
| `MoE_EP` | one full expert per rank | all-to-all dispatch + all-to-all combine |

On 4 ranks with one BLAS thread per rank, EP is about **1.6–2.3× faster** than TP for modest expert widths: each rank runs one expert on the tokens that routed to it, while TP still shards every active expert over the full batch. Details and tables are in [`moe/analysis.md`](moe/analysis.md).

```bash
source .venv/bin/activate
cd moe

mpirun -n 2 python test_moe.py
# hidden/output dims must divide the world size for TP (2, 5, and 10 work)

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mpirun -n 4 python benchmark.py --repeats 3
```

## Parallel primitives

`parallel/` is the communication layer underneath.

- **`myAllreduce` / `myAlltoall`** — reduce-to-root + broadcast, and pairwise `Sendrecv` all-to-all, with byte accounting. See [`parallel/discussion.md`](parallel/discussion.md).
- **`split_data`** — shard the dataset across data-parallel groups; model-parallel ranks in the same DP group share the split.
- **`get_info`** — MP-major rank layout, MP/DP communicators, and partitioned in/out dims for `fc_q/k/v` (output-sharded) and `fc_o` (input-sharded).
- **Naive model-parallel `W_o`** — all-gather on the forward path; slice + reduce-scatter on the backward path.

```bash
source .venv/bin/activate
cd parallel

mpirun -n 8 python mpi-test.py --test_case myallreduce
mpirun -n 8 python mpi-test.py --test_case myalltoall

python -m pytest -l -v tests/test_data_split.py
mpirun -n 8 python -m pytest -l -v --with-mpi tests/test_get_info.py
mpirun -n 4 python -m pytest -l -v --with-mpi tests/test_transformer_forward.py
mpirun -n 4 python -m pytest -l -v --with-mpi tests/test_transformer_backward.py
```

## Fused matmul kernel

[`kernels/matmul_triton.ipynb`](kernels/matmul_triton.ipynb) computes **`D = ReLU(A @ B + C)`** in fp16 with tiled shared-memory loads, a register accumulator, and fused add + ReLU. It is meant to be run on an NVIDIA GPU (Colab T4). The last cell grid-searches `BLOCK_M / BLOCK_N / BLOCK_K`.

Needs `torch` and `triton` in addition to `requirements.txt`.

## License

Code in this repository is provided for research and educational use.
