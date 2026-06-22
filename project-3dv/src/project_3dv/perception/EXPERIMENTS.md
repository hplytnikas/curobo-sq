# Experiments

Recipes for running the SuperDec test-split evaluation and the project unit tests.

---

## 1. LM fitter — CPU, no GPU required

Uses the classical Levenberg-Marquardt superquadric fitter.  Runs on any
login node without touching CUDA extensions.

```bash
cd /work/courses/3dv/team15/project-3dv

python3 scripts/test_on_superdec_split.py \
  --no-superdec \
  --n-samples 5 \
  --output-dir outputs/test_split_lm/
```

Outputs: `outputs/test_split_lm/sample_1.png` … `sample_5.png`

---

## 2. SuperDec neural fitter — GPU (interactive)

Requires a GPU node (PVCNN JIT-compiles CUDA extensions on first run,
then caches under `$TORCH_EXTENSIONS_DIR`).

```bash
srun -A 3dv -t 00:15:00 --gpus 1 \
  bash -c "
    source /work/courses/3dv/team15/superdec/.venv/bin/activate
    export PYTHONPATH=/work/courses/3dv/team15/superdec:\$PYTHONPATH
    export TORCH_EXTENSIONS_DIR=/work/courses/3dv/team15/.torch_extensions
    cd /work/courses/3dv/team15/project-3dv
    python3 scripts/test_on_superdec_split.py --n-samples 5 \
      --output-dir outputs/test_split_gpu/
  "
```

---

## 3. SuperDec neural fitter — GPU (SLURM batch)

Preferred for longer runs.  Submits a job via `run_gpu.sh`:

```bash
# 5 samples (default in run_gpu.sh):
sbatch scripts/run_gpu.sh

# Override sample count:
sbatch scripts/run_gpu.sh --n-samples 20 --output-dir outputs/test_split_20/

# Full test split (286 samples):
sbatch scripts/run_gpu.sh --n-samples 286 --output-dir outputs/test_split_full/
```

SLURM output: `/work/courses/3dv/team15/logs/test_split_<jobid>.out`

Track the job:
```bash
squeue --me
tail -f /work/courses/3dv/team15/logs/test_split_<jobid>.out
```

---

## 4. Output paths

| Run mode          | Output directory                              |
|-------------------|-----------------------------------------------|
| LM (CPU)          | `outputs/test_split_lm/`                      |
| SuperDec GPU      | `outputs/test_split_gpu/`                     |
| Full split (286)  | `outputs/test_split_full/`                    |

Each directory contains `sample_1.png` … `sample_N.png` (4-panel figures)
plus a summary table printed to stdout / the SLURM log.

---

## 5. Copy outputs to your local machine

```bash
# From your local machine:
scp -r \
  <username>@<cluster_hostname>:/work/courses/3dv/team15/project-3dv/outputs/ \
  ~/Downloads/3dv_outputs/
```

Or for a single file:
```bash
scp <username>@<cluster_hostname>:/work/courses/3dv/team15/project-3dv/outputs/test_split_gpu/sample_1.png .
```

---

## 6. Ablation — per-category canonical rotation

Ablation comparing SuperDec (fine-tuned, GPU) with and without
`SHAPENET_CATEGORY_ROTATIONS` applied, 3 samples per category, `orig-L2` metric.

| Category | with rotation | without rotation | verdict |
|----------|--------------|-----------------|---------|
| bottle   | ~0.15 (bad)  | **~0.074**      | NO rotation — model was fine-tuned on native ShapeNet Y-up; rotating away from it breaks inference |
| mug      | ~0.18 (bad)  | **~0.088**      | NO rotation — same reason |
| bowl     | **0.105**    | 0.115           | KEEP rotation — R_x(90) maps the bowl opening to face up correctly |

Conclusion applied in `SHAPENET_CATEGORY_ROTATIONS` (pipeline.py):
- `02876657` (bottle): `_r_x(0)` — identity
- `03797390` (mug): `_r_x(0)` — identity
- `02880940` (bowl): `_r_x(90)` — rotation kept

Also in this ablation cycle:
- `_filter_degenerate_primitives` threshold lowered 0.005 → 0.002 (0.005 was over-aggressive, causing zero-primitive outputs on some samples)
- Zero-primitive safety fallback added: if all primitives are filtered, the largest is kept

---

## 7. Unit tests

### Fast tests (no dataset required, ~75 tests)

```bash
cd /work/courses/3dv/team15/project-3dv
pytest tests/ -m "not requires_data" -v
```

### Dataset tests (requires OCID download)

```bash
bash scripts/download_ocid.sh          # downloads ~2 GB to data/ocid/
pytest tests/ -m requires_data -v
```

### All tests

```bash
pytest tests/ -v
```
