#!/usr/bin/env python3
"""
extract_data.py - rebuilds the data files of the paper from the raw result files.

Run from the repository root:
    python pipeline/aggregate.py --results results --out /tmp/agg
    python paper/scripts/extract_data.py --results results --agg /tmp/agg/all_runs.csv

Writes to paper/data/: all_runs.csv (without the processor column), models.csv, pfi.csv, environment.json.
The audit files (data_audit.csv, ambiguity_*.csv) come from pipeline/data_audit.py and need the dataset repository.
Processor utilization is dropped on purpose: the monitor used for these runs did not account for child processes.
"""
import argparse, glob, json, os, re
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--agg", required=True, help="all_runs.csv written by pipeline/aggregate.py")
ap.add_argument("--out", default="paper/data")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

runs = pd.read_csv(a.agg).drop(columns=["peak_cpu_percent"], errors="ignore")
runs.to_csv(os.path.join(a.out, "all_runs.csv"), index=False)

models, pfi, envs = [], [], set()
for f in sorted(glob.glob(os.path.join(a.results, "*.json"))):
    r = json.load(open(f))
    e = r["env"]
    envs.add(json.dumps({k: e[k] for k in ["os", "cpu_model", "cores_usable", "logical_cores", "physical_cores",
                                           "ram_gb", "python", "java", "packages"]}, sort_keys=True))
    fw = r["framework"]
    if fw.startswith("baseline"):
        continue
    mi = r["model_info"]
    tag = r["dataset"] + ("-raw" if r["split_info"].get("dedup") is False and r["dataset"].startswith("nbaiot") else "")
    row = dict(dataset=tag, split=r["split"], train_dist=r["train_dist"],
               framework=fw + ("-bal" if r.get("config", {}).get("h2o_balance_classes") else ""), seed=r["seed"])
    if fw == "h2o":
        row.update(leader_id=re.sub(r"_AutoML.*", "", mi["leader"]), leader_algo=mi["leader_algo"], n_members="", members="")
    else:
        row.update(leader_id="", leader_algo="", n_members=mi["n_ensemble_members"],
                   members=";".join(f"{m['estimator']}:{m['weight']}" for m in mi["members"]))
    models.append(row)
    if r.get("pfi"):
        for k, v in r["pfi"]["importance"].items():
            pfi.append(dict(framework=fw, seed=r["seed"], feature=k, importance=v))
pd.DataFrame(models).to_csv(os.path.join(a.out, "models.csv"), index=False)
pd.DataFrame(pfi).to_csv(os.path.join(a.out, "pfi.csv"), index=False)
assert len(envs) == 1, "results come from more than one environment"
json.dump(json.loads(next(iter(envs))), open(os.path.join(a.out, "environment.json"), "w"), indent=1)
print(f"wrote {len(runs)} runs, {len(models)} selected models, {len(pfi)} importance values")
