
#### `scripts/faiss_laptop_benchmark.py`
```python
#!/usr/bin/env python3
import time, math, argparse, os
import numpy as np
import faiss

def human_bytes(x):
    for unit in ["B","KB","MB","GB","TB","PB"]:
        if x < 1024: return f"{x:.2f} {unit}"
        x /= 1024
    return f"{x:.2f} EB"

def bench_flat(d=9472, nb=10000, nq=10, k=5, add_bs=2000, seed=42):
    print(f"\n[FlatL2] d={d}, nb={nb}, nq={nq}, k={k}, add_bs={add_bs}")
    rng = np.random.default_rng(seed)
    index = faiss.IndexFlatL2(d)
    print("Is trained:", index.is_trained)
    t0 = time.time()
    remain = nb
    while remain > 0:
        cur = min(add_bs, remain)
        xb = rng.random((cur, d), dtype=np.float32)
        index.add(xb)
        remain -= cur
    t_build = time.time() - t0
    print(f"Build/add time: {t_build:.2f}s")

    xq = rng.random((nq, d), dtype=np.float32)
    t0 = time.time()
    D, I = index.search(xq, k)
    t_search = time.time() - t0
    print(f"Search time: {t_search:.4f}s  ({t_search/nq*1000:.2f} ms/query)")
    print("Sample indices:", I[0])
    print("Sample distances:", D[0])
    bytes_est = nb * d * 4
    print("Approx index RAM (float32):", human_bytes(bytes_est))

def bench_ivfpq(d=9472, nb=100000, nq=10, k=5, nlist=4096, m=64, bits=8,
                nprobe=32, train_n=50000, bs=5000, seed=42):
    print(f"\n[IVF+PQ] d={d}, nb={nb}, nq={nq}, k={k}, nlist={nlist}, m={m}, bits={bits}, nprobe={nprobe}")
    rng = np.random.default_rng(seed)
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)

    train_n = min(train_n, nb)
    print(f"Preparing training set of {train_n} vectors…")
    t0 = time.time()
    trained = 0
    train_buf = []
    while trained < train_n:
        cur = min(bs, train_n - trained)
        train_buf.append(rng.random((cur, d), dtype=np.float32))
        trained += cur
    xtrain = np.vstack(train_buf)
    t_prep = time.time() - t0
    print(f"Training data prepared in {t_pep:.2f}s, shape={xtrain.shape}")  # typo fixed below

    print("Training IVF+PQ…")
    t0 = time.time()
    index.train(xtrain)
    t_train = time.time() - t0
    print(f"Train time: {t_train:.2f}s")

    print("Adding database…")
    t0 = time.time()
    added = 0
    while added < nb:
        cur = min(bs, nb - added)
        xb = rng.random((cur, d), dtype=np.float32)
        index.add(xb)
        added += cur
    t_add = time.time() - t0
    print(f"Add time: {t_add:.2f}s")

    index.nprobe = nprobe
    xq = rng.random((nq, d), dtype=np.float32)
    t0 = time.time()
    D, I = index.search(xq, k)
    t_search = time.time() - t0
    print(f"Search time: {t_search:.4f}s  ({t_search/nq*1000:.2f} ms/query)")
    print("Sample indices:", I[0])
    print("Sample distances:", D[0])

    pq_bytes = nb * m * (bits // 8)
    coarse_bytes = nlist * d * 4
    total_est = pq_bytes + coarse_bytes
    print("Approx index RAM (PQ codes + coarse):",
          human_bytes(total_est),
          f"(~{human_bytes(pq_bytes)} codes, ~{human_bytes(coarse_bytes)} centroids)")

if __name__ == "__main__":
    # Use all logical cores
    try:
        print("FAISS OMP threads (before):", faiss.omp_get_max_threads())
        faiss.omp_set_num_threads(os.cpu_count())
        print("FAISS OMP threads (after):", faiss.omp_get_max_threads())
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["flat","ivfpq"], default="flat")
    ap.add_argument("--d", type=int, default=9472)
    ap.add_argument("--nb", type=int, default=10000)
    ap.add_argument("--nq", type=int, default=10)
    ap.add_argument("--k",  type=int, default=5)
    ap.add_argument("--add_bs", type=int, default=2000)
    ap.add_argument("--nlist", type=int, default=4096)
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--train_n", type=int, default=50000)
    ap.add_argument("--bs", type=int, default=5000)
    args = ap.parse_args()

    if args.mode == "flat":
        bench_flat(d=args.d, nb=args.nb, nq=args.nq, k=args.k, add_bs=args.add_bs)
    else:
        # fix small variable name typo before running
        bench_ivfpq(d=args.d, nb=args.nb, nq=args.nq, k=args.k,
                    nlist=args.nlist, m=args.m, bits=args.bits,
                    nprobe=args.nprobe, train_n=args.train_n, bs=args.bs)
