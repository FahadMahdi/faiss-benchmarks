#!/usr/bin/env python3
import time, numpy as np, faiss, gc, os

d   = 9472
nb  = 100_000   # edit if needed
nq  = 10
k   = 5
bs  = 1000
seed = 42

try:
    print("FAISS OMP threads (before):", faiss.omp_get_max_threads())
    faiss.omp_set_num_threads(os.cpu_count())
    print("FAISS OMP threads (after):", faiss.omp_get_max_threads())
except Exception:
    pass

print(f"[FlatL2] d={d}, nb={nb}, nq={nq}, k={k}, add_bs={bs}")
rng = np.random.default_rng(seed)
index = faiss.IndexFlatL2(d)
print("Is trained:", index.is_trained)

xb_buf = np.empty((bs, d), dtype=np.float32)
t0 = time.time()
added = 0
while added < nb:
    cur = min(bs, nb - added)
    rng.random((cur, d), dtype=np.float32, out=xb_buf[:cur])
    index.add(xb_buf[:cur])
    added += cur
    if (added // bs) % 50 == 0:
        gc.collect()
t_add = time.time() - t0
print(f"Add time: {t_add:.2f}s")

xq = rng.random((nq, d), dtype=np.float32)
t0 = time.time()
D, I = index.search(xq, k)
t_search = time.time() - t0
print(f"Search: {t_search:.4f}s ({t_search/nq*1000:.2f} ms/query)")
print("Sample indices:", I[0])
print("Sample distances:", D[0])
