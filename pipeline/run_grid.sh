#!/usr/bin/env bash
# Experimental grid. Every cell uses the SAME seeds, split, budget and thread count for both frameworks.
#
# Tier 1 (main study):  BUDGET=1800 THREADS=8 SEEDS="0 1 2 3 4" DEVICES="7 1" ./run_grid.sh
# Tier 2 (more devices): BUDGET=1800 THREADS=8 SEEDS="0 1 2" DEVICES="2 3 4 5 6 8 9" MEMS=0 RAW=0 ./run_grid.sh
# Pilot (check everything first): SEEDS="0" DEVICES="7" ./run_grid.sh
# Preview without running: DRY=1 ./run_grid.sh        (prints the commands and the run count)
#
# Runs are independent: split SEEDS across machines/cores and share OUT. Finished runs are skipped,
# so the script can be restarted. Wall time ~= (#AutoML runs) x (BUDGET + ~60 s).
set -u
BUDGET=${BUDGET:-1800}; THREADS=${THREADS:-1}; SEEDS=${SEEDS:-"0 1 2 3 4"}
DEVICES=${DEVICES:-"7"}; OUT=${OUT:-results}; EXTRA=${EXTRA:-""}
MEMS=${MEMS:-1}; RAW=${RAW:-1}; DRY=${DRY:-0}
N=0
run() { N=$((N+1)); if [ "$DRY" = "1" ]; then echo "python run_automl.py --budget $BUDGET --threads $THREADS --out_dir $OUT $* $EXTRA";
        else python run_automl.py --budget "$BUDGET" --threads "$THREADS" --out_dir "$OUT" "$@" $EXTRA || echo "FAILED: $*"; fi; }
base() { if [ "$DRY" = "1" ]; then echo "python run_baselines.py $* --out_dir $OUT"; else python run_baselines.py "$@" --out_dir "$OUT"; fi; }

for s in $SEEDS; do
  if [ "$MEMS" = "1" ]; then
    base --dataset mems --split random --seed $s
    base --dataset mems --split group  --seed $s
    base --dataset mems --split group  --train_dist oversample --seed $s
    # (random,full) is comparable to XAI-IoT; (group,full) is leakage-aware;
    # (group,oversample) is the class-balancing ablation with identical data for both frameworks.
    for fw in autosklearn h2o; do
      run --framework $fw --dataset mems --split random --seed $s
      run --framework $fw --dataset mems --split group  --seed $s
      run --framework $fw --dataset mems --split group  --train_dist oversample --seed $s
    done
    run --framework h2o --dataset mems --split group --h2o_balance_classes --seed $s   # algorithmic balancing
  fi
  for d in $DEVICES; do
    base --dataset nbaiot:$d --seed $s
    [ "$RAW" = "1" ] && base --dataset nbaiot:$d --no_dedup --seed $s
    for fw in autosklearn h2o; do
      run --framework $fw --dataset nbaiot:$d --seed $s --pfi                          # deduplicated (default)
      [ "$RAW" = "1" ] && run --framework $fw --dataset nbaiot:$d --no_dedup --seed $s # raw, comparable to XAI-IoT
    done
  done
done
if [ "$DRY" = "1" ]; then
  echo "AutoML runs: $N  -> approx. $(( N * (BUDGET + 60) / 3600 )) hours sequential"
else
  python aggregate.py --results "$OUT" --out tables
fi
