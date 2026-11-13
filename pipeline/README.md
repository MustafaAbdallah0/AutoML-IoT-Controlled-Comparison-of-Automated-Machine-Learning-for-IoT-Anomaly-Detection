# AutoML for IoT anomaly detection - controlled comparison pipeline

Compares **Auto-sklearn** and **H2O AutoML** on the two datasets of the XAI-IoT repository
(MEMS vibration data, N-BaIoT botnet traffic), plus non-AutoML baselines. It is the successor of the
CICIDS/NSL-KDD pipeline, redesigned around the review comments on that paper.

## What the design fixes

| Review concern on the earlier paper | What the pipeline does |
|---|---|
| Frameworks trained on different data (capped vs full) | Both get the **same arrays**; training distribution (`full`, `cap:N`, `oversample`) is its own factor, applied to both |
| No ablation of class balancing | `--train_dist oversample`, plus `--h2o_balance_classes` as a separate algorithmic factor |
| Unequal budgets (30 vs 15 min) and threads (H2O all cores, ASK 1) | One `--budget` and one `--threads` (H2O `nthreads` = ASK `n_jobs`) |
| Single runs, no variance | `--seed` loop in `run_grid.sh`; `aggregate.py` reports mean +/- std |
| No hardware/software record | `env_info.py` stores CPU, cores, RAM, OS, Java, package versions **in every result file** |
| Split leakage / optimistic scores | exact duplicates + label conflicts removed (N-BaIoT); `--split group` holds out whole MEMS runs |
| "H2O optimises accuracy" claims | both use class-balanced objectives (ASK `f1_macro`, H2O `mean_per_class_error`); reported explicitly |
| AutoML vs simple models unclear | `run_baselines.py`: dummy, decision tree, random forest, MLP on the identical split |
| Overhead table not comparable | peak RSS/CPU (process tree incl. JVM), fit time, models evaluated, **model size and inference latency** (edge relevance) |

## Quick start
```bash
git clone https://github.com/agummadi1/XAI_for_IoT_Systems.git
export IOT_DATA_DIR=$PWD/XAI_for_IoT_Systems/Datasets
python data_audit.py                                   # dataset facts -> tables/table_audit.tex
python run_baselines.py --dataset mems --split group --seed 0
python run_automl.py --framework h2o --dataset mems --split group --budget 300 --seed 0 --pfi
python run_automl.py --framework autosklearn --dataset mems --split group --budget 300 --seed 0
python aggregate.py                                    # -> tables/*.tex, summary_mean_std.csv
DRY=1 SEEDS="0" DEVICES="7" ./run_grid.sh                               # preview commands + run count
SEEDS="0" DEVICES="7" ./run_grid.sh                                      # pilot (11 AutoML runs, ~5.7 h at 1800 s)
```

## Staged plan (budget 1800 s per run, same THREADS for both frameworks)
| Stage | Command | AutoML runs | Sequential time |
|---|---|---|---|
| Pilot | `SEEDS="0" DEVICES="7" ./run_grid.sh` | 11 | ~5.7 h |
| Tier 1 (main) | `THREADS=8 SEEDS="0 1 2 3 4" DEVICES="7 1" ./run_grid.sh` | 75 | ~38.8 h |
| Tier 2 (more devices) | `THREADS=8 SEEDS="0 1 2" DEVICES="2 3 4 5 6 8 9" MEMS=0 RAW=0 ./run_grid.sh` | 42 | ~21.7 h |

Run the pilot first and inspect `tables/` before committing to Tier 1. Runs are independent and resumable: give each
machine/core its own `SEEDS` and the same `OUT`, then run `python aggregate.py` once. Use the same `THREADS` for every run.

## Scope of the main study: MEMS + N-BaIoT device 9
```bash
THREADS=8 SEEDS="0" DEVICES="9" ./run_grid.sh            # pilot: 11 AutoML runs, ~5.7 h at 1800 s
THREADS=8 SEEDS="0 1 2 3 4" DEVICES="9" ./run_grid.sh    # full: 55 AutoML runs, ~28.5 h sequential
```
Per seed the grid writes 31 result files (20 baseline + 11 AutoML), i.e. 155 for five seeds. Raw (duplicated) N-BaIoT results carry a
`-raw` suffix (`nbaiot9-raw__...`), deduplicated ones do not. Device 9 after cleaning (seed 0, 70/30): train 81,951, test 35,122,
11 classes; `class_3` and `class_4` have fewer than 50 unique rows and only 3 and 5 test samples (`class_1`: 26), so their recall is indicative only.

## Safety against double launches and parallel runs
* `run_automl.py` takes a per-cell lock (`<result>.json.lock`); a second process on the same cell exits with "already running",
  and a lock left by a dead process is taken over.
* Each H2O run starts its own cluster on a free port, so several runs can share a machine (tested with two concurrent runs).
* Do not start `run_grid.sh` twice with the same seeds/devices; use `tmux` or a scheduler job.

## Analysis scripts and fixes (after the first full study)
* `python data_audit.py` now also writes `ambiguity_summary.csv` / `ambiguity_device9.csv`: feature vectors that occur under more than one
  label, the share of rows they cover, and the accuracy ceiling they imply (computed on the raw file).
* `python make_paper_tables.py --runs tables/all_runs.csv --amb tables --results results --out tables` (add `--results` for the permutation-importance table `paper_pfi.tex` and `pfi_summary.csv`) generates the compact paper tables
  (`paper_mems.tex`, `paper_dev9.tex`, `paper_paired.tex`, `paper_perclass9.tex`, `paper_resources.tex`, `paper_ambiguity.tex`).
* **Resource monitor fixed.** The first version called psutil `cpu_percent(interval=None)` on freshly created `Process` objects every loop,
  which returns 0.0 on first use, so CPU of all child processes (Auto-sklearn workers, the H2O JVM) was lost: a busy child process was
  reported as 2% instead of ~100%. `peak_cpu_percent` in results produced before this fix is **not valid** (memory numbers are unaffected).
  New results also store `mean_cpu_percent`. Test: run a busy child process under the monitor and check it reports ~100%.
* To obtain valid CPU numbers without repeating the study, re-run a few cells into a separate folder, e.g.
  `python run_automl.py --framework h2o --dataset nbaiot:9 --seed 0 --budget 1800 --threads 8 --pfi --out_dir results_cpu`.

## Decisions recorded
* Budget: 1800 s per run for both frameworks.
* Class ids are reported as `class_<id>` (names unverified; `IOT_USE_NAMES=1` switches on the assumed names).
* MEMS true session/RPM ids are not available: the inferred label-block groups are used and listed as a limitation.
* Only the released 20-feature N-BaIoT subset is evaluated; the original 115-feature data may be added later.
Dataset names: `mems`, `nbaiot:<1-9>`. Set up the Python 3.9 environment as in `requirements_ask_env.txt`.

## Data caveats found while building this (please verify with the dataset owners)
1. **N-BaIoT files are heavily duplicated**: 42-48% exact duplicate rows on every device; each device has
   2-3 classes with fewer than 50 unique rows (as few as 9) repeated up to 20,000 times
   (`tables/table_audit.tex`). A random split puts copies in train and test. The default loader therefore deduplicates;
   `--no_dedup` reproduces the XAI-IoT setting.
2. The repo holds only the **top-20 features** per device, not the 115 original N-BaIoT features.
3. **MEMS has no session/RPM column.** Run ids are *inferred* from contiguous label blocks in file order
   (15 blocks, median 1,423 rows). Random-split RF: accuracy 0.71 / macro-F1 0.66; run-grouped 5-fold: 0.54 / 0.35.
   If the real RPM/session ids exist, pass them in place of the inferred runs.
4. **Class names are unverified.** The CSVs contain integer ids; the id->name mapping from the XAI-IoT text is used only with
   `IOT_USE_NAMES=1`. Default output uses `class_<id>`.
5. `time` in the MEMS file is a constant (`6-30`) and is dropped.

## Tested here (1 core, 3 GB RAM, Python 3.9.25, auto-sklearn 0.15.0, scikit-learn 0.24.2, h2o 3.46.0.12)
Data loading/splits/audit; all four baselines; H2O and Auto-sklearn end-to-end on MEMS (group split, with `--pfi`)
and on N-BaIoT device 7 (subsampled; H2O with `--h2o_balance_classes`, Auto-sklearn with `--ask_refit`); aggregation.
Runs used 45-90 s budgets, so the scores are **plumbing checks, not results**.
**Not tested:** multi-core runs, budgets >= 300 s, Colab, the full grid, `--train_dist cap:N` with the frameworks
(split logic is tested).

## Known limitations of the protocol
* `wall_clock_fit_s` includes JVM start and data upload for H2O (a few seconds beyond `--budget`).
* H2O validates by 5-fold CV and ASK by 67/33 holdout (its default); use `--ask_refit` to refit ASK on all training data.
* Wall-clock-limited searches are not exactly reproducible even with a fixed seed (use `--h2o_max_models` for H2O).
* Auto-sklearn already searches over a class-weighting strategy internally; `oversample` is the common data-level control.
* `n_models_evaluated` counts H2O leaderboard rows vs Auto-sklearn run-history entries: not the same quantity.
