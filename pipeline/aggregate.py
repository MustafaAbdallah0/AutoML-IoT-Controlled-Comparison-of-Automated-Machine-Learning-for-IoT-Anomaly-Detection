"""aggregate.py - mean +/- std over seeds -> CSV and LaTeX tables (booktabs).

  python aggregate.py --results results --out tables
Groups by (dataset, split, train_dist, framework). Std needs >= 2 seeds.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

MAIN = [("macro_f1", "Macro-F1"), ("balanced_accuracy", "Bal. acc."), ("mcc", "MCC"), ("accuracy", "Acc.")]
RES = [("wall_clock_fit_s", "Fit (s)"), ("peak_rss_mb", "Peak mem (MB)"), ("n_models_evaluated", "Models eval."),
       ("model_size_mb", "Model (MB)"), ("predict_latency_ms_per_sample", "Latency (ms/sample)")]


def load(results):
    rows = []
    for f in sorted(glob.glob(os.path.join(results, "*.json"))):
        r = json.load(open(f))
        raw = r["dataset"].startswith("nbaiot") and r.get("split_info", {}).get("dedup") is False
        row = {"dataset": r["dataset"] + ("-raw" if raw else ""), "split": r["split"], "train_dist": r["train_dist"],
               "framework": r["framework"] + ("-bal" if r.get("config", {}).get("h2o_balance_classes") else ""),
               "seed": r["seed"]}
        row.update(r["metrics"])
        row.update(r["resources"])
        for c, v in r["per_class"].items():
            row[f"recall::{c}"] = v["recall"]
        rows.append(row)
    return pd.DataFrame(rows)


def pm(s, nd=3):
    s = s.dropna()
    if len(s) == 0:
        return "--"
    if len(s) == 1:
        return f"{s.iloc[0]:.{nd}f}"
    return f"{s.mean():.{nd}f} $\\pm$ {s.std(ddof=1):.{nd}f}"


def table(df, cols, caption, label, nd=3):
    keys = ["dataset", "split", "train_dist"]
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\scriptsize",
             "\\begin{tabular}{@{}lllr" + "r" * len(cols) + "@{}}", "\\toprule",
             "Dataset & Split & Framework & Seeds & " + " & ".join(c[1] for c in cols) + " \\\\", "\\midrule"]
    for k, g in df.groupby(keys, sort=True):
        first = True
        for fw, h in g.groupby("framework", sort=True):
            ds = f"{k[0]}" if first else ""
            sp = f"{k[1]}/{k[2]}" if first else ""
            nds = [3 if c[0] in ("macro_f1", "balanced_accuracy", "mcc", "accuracy", "predict_latency_ms_per_sample") else 1
                   for c in cols]
            vals = " & ".join(pm(h[c[0]], n) if c[0] in h else "--" for c, n in zip(cols, nds))
            lines.append(f"{ds} & {sp} & {fw.replace('_', '-')} & {h['seed'].nunique()} & {vals} \\\\")
            first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="tables")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load(a.results)
    if df.empty:
        raise SystemExit("no result files found")
    df.to_csv(os.path.join(a.out, "all_runs.csv"), index=False)
    summ = df.groupby(["dataset", "split", "train_dist", "framework"]).agg(
        ["mean", "std"], numeric_only=True) if False else \
        df.drop(columns=["seed"]).groupby(["dataset", "split", "train_dist", "framework"]).agg(["mean", "std", "count"])
    summ.to_csv(os.path.join(a.out, "summary_mean_std.csv"))
    open(os.path.join(a.out, "table_quality.tex"), "w").write(table(
        df, MAIN, "Detection quality (mean $\\pm$ std over seeds).", "tab:quality"))
    open(os.path.join(a.out, "table_resources.tex"), "w").write(table(
        df, RES, "Resource use and deployment cost.", "tab:resources"))
    rc = [c for c in df.columns if c.startswith("recall::")]
    pcols = [(c, c.split("::")[1]) for c in rc]
    open(os.path.join(a.out, "table_per_class_recall.tex"), "w").write(table(
        df, pcols, "Per-class recall.", "tab:perclass"))
    print(f"{len(df)} runs -> {a.out}/ (all_runs.csv, summary_mean_std.csv, table_*.tex)")


if __name__ == "__main__":
    main()
