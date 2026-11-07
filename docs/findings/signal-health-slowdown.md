# signal_health in the oneDAL process

On the 2-vCPU Xeon in `results/intel-xeon-2t/summary.md`, plant M `signal_health` is slower in the oneDAL process (109 ms stock, 122 ms oneDAL) even though every PCA fit in the parity log stayed on stock. The suspicion was thread-pool contention between oneTBB and the BLAS library on a machine with few cores.

This checkout is a 12-thread Windows laptop, not that Xeon. The same tool is faster here, not slower.

## What this machine measured

`results/_scratch/summary.md`, plant M, seed 7, median of 3 warm on-demand calls, one line:

| | Stock ms | oneDAL ms | Speed-up |
|---|---:|---:|---:|
| signal_health | 440 | 284 | 1.55× |

Nightly, whole plant, warm: 1.97 s stock, 1.55 s oneDAL (1.27×).

A separate probe, `results/findings/signal-health-slowdown.json`, times **stock** `signal_health` on line L1 of plant M seed 7. Each case is a fresh process, one warm-up and 5 calls, median reported. `import sklearnex` happens before the tools are imported. `after_rf` also fits a 50-tree oneDAL forest on 2,000 rows first. The `threads1_*` cases set OMP, MKL, OpenBLAS, TBB and DAAL thread variables to 1 before the process starts.

| Case | Median ms | Pools threadpoolctl reported |
|---|---:|---|
| stock | 328.95 | OpenBLAS 12, MSVC OpenMP (`vcomp`) 12 |
| import sklearnex | 311.34 | same, 12 |
| after a oneDAL forest fit | 310.63 | same, 12 |
| threads = 1, stock | 305.2 | OpenBLAS 1, vcomp 1 |
| threads = 1, sklearnex imported | 205.05 | OpenBLAS 1, vcomp 1 |
| threads = 1, after the forest fit | 217.31 | OpenBLAS 1, vcomp 1 |

threadpoolctl listed no TBB pool in any case. Importing scikit-learn-intelex did not slow the stock tool on this CPU. Limiting the visible pools to 1 thread and then importing sklearnex made the median lower (305.2 ms to 205.05 ms), still with only OpenBLAS and vcomp visible. This probe does not explain that drop.

## Conclusion

The 5–18% slowdown from the Xeon note was **not found** on this machine. No tool change. Re-run the probe on a 2-thread host before treating TBB-versus-BLAS contention as the cause.
