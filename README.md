# AutoML-IoT

Code, data, and paper source for **"AutoML-IoT: Controlled Comparison of Automated Machine Learning for IoT Anomaly Detection"**.

The study compares Auto-sklearn and H2O AutoML with default baselines on a sensor dataset (MEMS) and a botnet-traffic dataset (N-BaIoT, device 9) under identical data, budgets, threads, and seeds. It audits the data, constructs leakage-aware protocols, and measures quality, cost, and feature reliance.

## Repository layout

| Folder | Content |
|---|---|
| `pipeline/` | Experiment code: data loading and audit, controlled runs of both frameworks and the baselines, aggregation (`run_grid.sh` is the entry point) |
| `results/` | The 155 raw result files of the published study (one JSON per run) |
| `notebooks/` | `run_study.ipynb` starts and monitors the study as background processes |
| `paper/` | LaTeX source of the paper, the data files its tables are generated from, and the table scripts |

## Quick start

**1. Environment.** The published runs used Python 3.9 and the pinned packages in `requirements.txt`. Auto-sklearn needs `swig`, and H2O needs a Java runtime (OpenJDK 17 was used). The exact installation steps are in `pipeline/requirements_ask_env.txt`.

**2. Datasets.** The datasets are not redistributed here. Clone the repository of the XAI-IoT study and point the code to it:
```bash
git clone https://github.com/agummadi1/XAI_for_IoT_Systems.git
export IOT_DATA_DIR=$(pwd)/XAI_for_IoT_Systems/Datasets
```

**3. Re-run the study** (about 28.5 hours sequentially: 55 AutoML runs of about 31 minutes each).
```bash
cd pipeline
THREADS=8 SEEDS="0 1 2 3 4" DEVICES="9" OUT=results ./run_grid.sh
```
Use `DRY=1` to preview the commands. Finished runs are skipped, so an interrupted study can be resumed with the same command. Do not start two copies with the same seeds.

**4. Rebuild the tables and the paper data from the published results.**
```bash
python pipeline/aggregate.py --results results --out /tmp/agg
python paper/scripts/extract_data.py --results results --agg /tmp/agg/all_runs.csv
cd paper && python scripts/make_tables.py
```

**5. Build the paper.** Upload `paper/` to Overleaf, or run `pdflatex main`, `bibtex main`, `pdflatex main` twice inside `paper/`.

## Result files

Each file in `results/` is named `<dataset>__<split>__<train_dist>__<framework>__s<seed>.json` and stores the seed, the configuration, the hardware and package versions (`env`), the quality metrics (`metrics`, including per-class recall), the cost measurements (`resources`), and the selected model (`model_info`). Runs of device 9 on the raw file carry the suffix `-raw` in the dataset name, and permutation importance (`pfi`) is stored for the deduplicated device-9 runs of both frameworks.

## Published environment

Intel Core i9-10900X (10 cores, 20 threads), 156.9 GB RAM, Linux 6.8.0, Python 3.9.25, Auto-sklearn 0.15.0, H2O 3.46.0.12, scikit-learn 0.24.2, NumPy 1.23.5, pandas 1.5.3, SciPy 1.10.1, OpenJDK 17.0.18. All 155 runs share this environment, and all 55 AutoML runs share one configuration (1800 s budget, 8 threads, 4 GB H2O heap, 3072 MB per Auto-sklearn run).

## Known issues and notes

- **Processor utilization is not valid in the published results.** The resource monitor used for these runs did not account for child processes, so the field `peak_cpu_percent` must not be used. The paper does not report it. The monitor in `pipeline/resource_monitor.py` is fixed and also records `mean_cpu_percent`. Peak memory, fit time, model size, and latency are not affected.
- **Anonymized paths.** The local data path in the configuration of the AutoML result files was replaced by `XAI_for_IoT_Systems/Datasets`.
- **MEMS sessions are inferred.** The released file has no session or speed column, so blocks of identical labels serve as an inferred grouping.
- **Class names.** The integer labels follow the class order documented for the XAI-IoT datasets (MEMS: 1 normal, 2 near-failure, 3 failure; N-BaIoT: 1 benign, 2 to 6 Gafgyt, 7 to 11 Mirai). Confirm the mapping with the dataset authors if it is critical.
- **Reproducibility.** Searches that stop on a wall-clock limit are not exactly reproducible, even with fixed seeds, so a re-run gives similar but not identical numbers.

## License

Code: MIT (see `LICENSE`). Choose the license of the result files and check that it is compatible with the license of the XAI-IoT datasets from which they are derived.

## Citation

TODO: add the BibTeX entry of the paper after publication.

## Acknowledgments

The datasets and the XAI-IoT study are due to Gummadi, Napier, and Abdallah (IEEE Access, 2024).
