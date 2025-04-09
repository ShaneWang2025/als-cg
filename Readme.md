# Parallel ALS-CG with MPI

This project implements a parallelized ALS algorithm with a Conjugate Gradient (CG) solver using MPI.  
It supports collaborative filtering on large-scale rating data using sparse matrix representation.

---

## Directory Structure

```bash
.
├── als-cg_parallel.c      # Main C implementation
├── compile.sh             # Shell script to compile the program
├── submit_job.sh          # Example PBS job submission script
├── qsub_als               # qsub script
├── rating/                # Input dataset folder (CSV). Due to size of file, only contain 100k and 1m.
│   ├── ml-100k.csv
│   ├── ml-1m.csv
│   ├── ml-10m.csv
│   ├── ml-20m.csv
│   └── ml-32m.csv
