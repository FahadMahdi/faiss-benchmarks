---

# FAISS Benchmarks (9472-D)

Short, reproducible benchmarks for **exact** and **approximate** nearest-neighbor search using **FAISS**. Focus: **9,472-dim** embeddings at **10k–100k** scale, tested on a Windows laptop (CPU) and compared to WS1.

---

## ⚙️ Requirements

* **Python** 3.11–3.12 (tested on 3.12)
* **Pip**
* **FAISS (CPU)** and **NumPy**

Install the Python deps:

```bash
# Windows (PowerShell)
py -m pip install -r requirements.txt

# Linux/macOS
python3 -m pip install -r requirements.txt
```

`requirements.txt`

```
numpy==2.3.2
faiss-cpu==1.12.0
```

> GPU is **not** required for these CPU benchmarks.

---

## 📁 Repo structure

```
.
├─ README.md
├─ requirements.txt
└─ scripts/
   ├─ faiss_laptop_benchmark.py   # exact (FlatL2) + IVF+PQ benchmark driver
   └─ faiss_flat_lowmem.py        # exact FlatL2 with minimal RAM overhead
```

---

## 🔌 Power & Threads (important for laptops)

* Plug in the laptop and set **Power mode: Best performance**.
* Scripts pin FAISS to all logical cores:

  ```python
  import os, faiss
  faiss.omp_set_num_threads(os.cpu_count())
  ```

Optional (set once in a terminal if you see weirdly slow runs):

```bash
# Windows (PowerShell)
setx OMP_NUM_THREADS 16
setx MKL_NUM_THREADS 16
# re-open terminal after setting
```

---

## 🚀 Quickstart

### Exact search (IndexFlatL2)

Small, safe run:

```bash
# Windows
py scripts/faiss_laptop_benchmark.py --mode flat --d 9472 --nb 10000 --nq 10 --k 5 --add_bs 2000

# Linux/macOS
python3 scripts/faiss_laptop_benchmark.py --mode flat --d 9472 --nb 10000 --nq 10 --k 5 --add_bs 2000
```

Larger (needs RAM; scales linearly with `nb`):

```bash
py scripts/faiss_laptop_benchmark.py --mode flat --d 9472 --nb 50000 --nq 10 --k 5 --add_bs 2000
py scripts/faiss_laptop_benchmark.py --mode flat --d 9472 --nb 60000 --nq 10 --k 5 --add_bs 2000
```

**100k exact** may fail on 16 GB laptops (needs \~3.6–3.8 GB just for the index). If you want to try anyway:

* Reboot, close apps, pause OneDrive/Dropbox, increase pagefile; or
* Use `scripts/faiss_flat_lowmem.py` (pre-allocates a single buffer), then edit `nb` inside the file:

  ```bash
  py scripts/faiss_flat_lowmem.py
  ```

### Approximate search (IVF+PQ) — full scale on low RAM

Run 100k × 9472 with tiny memory:

```bash
py scripts/faiss_laptop_benchmark.py --mode ivfpq --d 9472 --nb 100000 ^
  --train_n 50000 --nlist 4096 --m 64 --bits 8 --nprobe 32 --bs 5000
```

* **RAM est.** ≈ codes `(nb*m*1B)` + centroids `(nlist*d*4B)`
  → \~**6.4 MB + 148 MB ≈ 154 MB**
* **Tuning**:

  * Increase `--nprobe 64`/`128` → higher recall, slower
  * Increase `--nlist 8192` → finer partitioning (train ↑, memory ↑)

---

## 🔎 CLI Arguments (common)

* `--mode {flat,ivfpq}`: exact vs IVF+PQ
* `--d`: embedding dimension (here **9472**)
* `--nb`: database size (10k–100k)
* `--nq`: number of queries
* `--k`: top-k neighbors
* `--add_bs` / `--bs`: chunk size for adding vectors (RAM control)

**IVF+PQ-only**

* `--nlist`: number of coarse clusters
* `--m`: PQ sub-vectors (ideally divides `d`, FAISS can pad otherwise)
* `--bits`: bits per sub-quantizer (8 is common)
* `--nprobe`: clusters to probe per query (speed/recall trade-off)
* `--train_n`: number of vectors for training (use a representative subset)

---

## 🧮 Memory math (sanity check)

Exact (FlatL2) stores raw float32 vectors:

```
RAM ≈ nb × d × 4 bytes
d = 9472
10k  → ~361 MB
50k  → ~1.76 GB
60k  → ~2.12 GB
100k → ~3.53 GiB (~3.8 GB)
```

IVF+PQ stores compressed codes + coarse centroids (see estimate above).

---

