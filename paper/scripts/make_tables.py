#!/usr/bin/env python3
"""
make_tables.py - generates every result table of the paper from the data files in ../data.

    python scripts/make_tables.py          (run from the project root)

Input  (data/): all_runs.csv, data_audit.csv, ambiguity_summary.csv, ambiguity_device9.csv,
                models.csv, pfi.csv
Output (tables/): table_*.tex

Bold convention used in all tables: the best value of a column (highest quality, lowest cost).
Ties at the printed precision are all bold, nothing is bold when all entries tie, and the
majority-class predictor is never ranked. Paired-difference cells are bold when the first method
won all five seeds.
"""
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = lambda f: os.path.join(ROOT, "data", f)
T = lambda f: os.path.join(ROOT, "tables", f)

NAME = {"autosklearn": "Auto-sklearn", "h2o": "H2O AutoML", "h2o-bal": "H2O AutoML, balanced",
        "baseline-rf": "Random forest", "baseline-mlp": "MLP", "baseline-dt": "Decision tree",
        "baseline-dummy": "Majority class"}
ORDER = ["autosklearn", "h2o", "h2o-bal", "baseline-rf", "baseline-mlp", "baseline-dt", "baseline-dummy"]
RANKED = [m for m in ORDER if m != "baseline-dummy"]
SETUPS = [("mems|random|full", "MEMS, random split"), ("mems|group|full", "MEMS, block split"),
          ("mems|group|oversample", "MEMS, block split with oversampling"),
          ("nbaiot:9|random|full", "Device 9, deduplicated"), ("nbaiot:9-raw|random|full", "Device 9, raw")]
SEP8 = [f"recall::class_{i}" for i in (2, 5, 6, 7, 8, 9, 10, 11)]


def bold(text, on):
    return f"\\textbf{{{text}}}" if on else text


def best_set(values, mode="max"):
    vals = {k: v for k, v in values.items() if v is not None and not np.isnan(v)}
    if len(vals) < 2 or len(set(vals.values())) == 1:
        return set()
    target = max(vals.values()) if mode == "max" else min(vals.values())
    return {k for k, v in vals.items() if v == target}


def wrap(body, caption, label, cols, star=True, size="scriptsize", tabcolsep=3):
    env = "table*" if star else "table"
    return (f"\\begin{{{env}}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\{size}\n"
            f"\\setlength{{\\tabcolsep}}{{{tabcolsep}pt}}\n\\begin{{tabular}}{{{cols}}}\n\\toprule\n{body}\n"
            f"\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n")


def write(name, text):
    with open(T(name), "w") as f:
        f.write(text)


def esc(s):
    return str(s).replace("_", "\\_").replace("%", "\\%")


df = pd.read_csv(D("all_runs.csv"))
df["cond"] = df.dataset + "|" + df.split + "|" + df.train_dist
df["sep8"] = df[SEP8].mean(axis=1)
sel = lambda cond, fw, col: df[(df.cond == cond) & (df.framework == fw)][col].dropna()


def ms(cond, fw, col, nd=3):
    s = sel(cond, fw, col)
    return None if len(s) == 0 else (s.mean(), s.std(ddof=1) if len(s) > 1 else 0.0)


def cell(v, nd=3, on=False):
    return "--" if v is None else bold(f"{v[0]:.{nd}f} $\\pm$ {v[1]:.{nd}f}", on)


# ------------------------------------------------------------------------- audit (combined)
audit = pd.read_csv(D("data_audit.csv")).merge(pd.read_csv(D("ambiguity_summary.csv")).drop(columns=["rows_raw"]), on="dataset")
nb = audit[audit.dataset.str.startswith("nbaiot")]
bests = {"duplicate_pct": set(nb[nb.duplicate_pct == nb.duplicate_pct.min()].dataset),
         "min_unique_per_class": set(nb[nb.min_unique_per_class == nb.min_unique_per_class.max()].dataset),
         "ambiguous_vectors": set(nb[nb.ambiguous_vectors == nb.ambiguous_vectors.min()].dataset),
         "rows_in_ambiguous_vectors_pct": set(nb[nb.rows_in_ambiguous_vectors_pct == nb.rows_in_ambiguous_vectors_pct.min()].dataset),
         "ceiling_accuracy": set(nb[nb.ceiling_accuracy.round(3) == nb.ceiling_accuracy.round(3).max()].dataset)}
rows = ["Dataset & Classes & Rows & Unique rows & Removed (\\%) & Smallest class & Ambiguous vectors & Rows in them (\\%) & Accuracy ceiling \\\\",
        "\\midrule"]
for _, r in audit.iterrows():
    b = lambda col, txt: bold(txt, r.dataset in bests[col])
    rows.append(f"{r.dataset} & {r.classes} & {int(r.rows_raw):,} & {int(r.rows_unique):,} & "
                f"{b('duplicate_pct', f'{r.duplicate_pct:.1f}')} & {b('min_unique_per_class', f'{int(r.min_unique_per_class):,}')} & "
                f"{b('ambiguous_vectors', str(int(r.ambiguous_vectors)))} & "
                f"{b('rows_in_ambiguous_vectors_pct', f'{r.rows_in_ambiguous_vectors_pct:.1f}')} & "
                f"{b('ceiling_accuracy', f'{r.ceiling_accuracy:.3f}')} \\\\")
    if r.dataset == "mems":
        rows.append("\\midrule")
write("table_audit.tex", wrap("\n".join(rows),
      "Audit of the released datasets. Bold: best N-BaIoT value per column. Smallest class: unique rows of the smallest class after cleaning.",
      "tab:audit", "@{}lrrrrrrrr@{}"))

# ------------------------------------------------------------------------- ambiguity per class (device 9)
am = pd.read_csv(D("ambiguity_device9.csv"))
rows = ["Class & Rows & Unique vectors & Rows in repeated vectors (\\%) & Rows in ambiguous vectors (\\%) & Ceiling recall \\\\", "\\midrule"]
for _, r in am.iterrows():
    rows.append(f"{r['class'].replace('class_', 'c')} & {int(r.raw_rows):,} & {int(r.unique_vectors):,} & {r.duplicate_pct:.1f} & "
                f"{r.rows_in_ambiguous_vectors_pct:.1f} & {r.ceiling_recall:.3f} \\\\")
write("table_amb_class9.tex", wrap("\n".join(rows),
      "Per-class audit of N-BaIoT device 9. Ceiling recall: share of the class that a majority-label rule classifies correctly.",
      "tab:ambclass", "@{}lrrrrr@{}", star=True))

# ------------------------------------------------------------------------- MEMS main table
fws = ["autosklearn", "h2o", "h2o-bal", "baseline-rf", "baseline-mlp", "baseline-dt", "baseline-dummy"]
cols = [("mems|random|full", "macro_f1"), ("mems|group|full", "macro_f1"), ("mems|group|oversample", "macro_f1"),
        ("mems|group|full", "balanced_accuracy")]
vals = {(fw, j): ms(c, fw, m) for fw in fws for j, (c, m) in enumerate(cols)}
rows = ["Method & Macro-F1, random split & Macro-F1, block split & Macro-F1, block + oversampling & Balanced accuracy, block split \\\\", "\\midrule"]
for fw in fws:
    cs = []
    for j in range(4):
        win = best_set({f: (round(vals[(f, j)][0], 3) if vals[(f, j)] else None) for f in RANKED})
        cs.append(cell(vals[(fw, j)], 3, fw in win))
    rows.append(f"{NAME[fw]} & " + " & ".join(cs) + " \\\\")
write("table_mems.tex", wrap("\n".join(rows),
      "MEMS: mean $\\pm$ standard deviation over five seeds for three protocols. Bold: highest mean per column.",
      "tab:mems", "@{}lrrrr@{}"))

# ------------------------------------------------------------------------- device 9 main table
d, r_ = "nbaiot:9|random|full", "nbaiot:9-raw|random|full"
fw9 = ["autosklearn", "h2o", "baseline-rf", "baseline-mlp", "baseline-dt", "baseline-dummy"]
cols9 = [(d, "macro_f1"), (d, "accuracy"), (d, "sep8"), (r_, "macro_f1"), (r_, "accuracy")]
v9 = {(fw, j): ms(c, fw, m) for fw in fw9 for j, (c, m) in enumerate(cols9)}
rows = ["Method & \\multicolumn{3}{c}{Deduplicated} & \\multicolumn{2}{c}{Raw} \\\\", "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}",
        " & Macro-F1 & Accuracy & Recall, 8 classes & Macro-F1 & Accuracy \\\\", "\\midrule"]
for fw in fw9:
    cs = []
    for j in range(5):
        win = best_set({f: (round(v9[(f, j)][0], 3) if v9[(f, j)] else None) for f in fw9 if f != "baseline-dummy"})
        cs.append(cell(v9[(fw, j)], 3, fw in win))
    rows.append(f"{NAME[fw]} & " + " & ".join(cs) + " \\\\")
amb9 = pd.read_csv(D("ambiguity_summary.csv")).query("dataset == 'nbaiot:9'").iloc[0]
rows += ["\\midrule", f"Accuracy ceiling (raw) & -- & -- & -- & -- & {amb9.ceiling_accuracy:.3f} \\\\"]
write("table_dev9.tex", wrap("\n".join(rows),
      "Device 9: mean $\\pm$ standard deviation over five seeds. Recall, 8 classes: mean recall over classes 2 and 5--11. Bold: highest mean per column.",
      "tab:dev9", "@{}lrrrrr@{}"))

# ------------------------------------------------------------------------- per-class recall, device 9 (main: three methods)
cls = [f"recall::class_{i}" for i in range(1, 12)]
rows = ["Variant & Method & " + " & ".join(f"c{i}" for i in range(1, 12)) + " \\\\", "\\midrule"]
for cond, lab, amb in [(d, "Deduplicated", set()), (r_, "Raw", {1, 3, 4})]:
    means = {fw: df[(df.cond == cond) & (df.framework == fw)][cls].mean().values for fw in ["autosklearn", "h2o", "baseline-rf"]}
    wins = [best_set({fw: round(means[fw][i], 2) for fw in means}) if (i + 1) not in amb else set() for i in range(11)]
    for fw in means:
        rows.append(f"{lab} & {NAME[fw]} & " + " & ".join(bold(f"{means[fw][i]:.2f}", fw in wins[i]) for i in range(11)) + " \\\\")
    rows.append("\\midrule")
rows.pop()
write("table_perclass9.tex", wrap("\n".join(rows),
      "Device 9: mean per-class recall over five seeds. Bold: best method per class and variant; raw c1, c3, c4 are not ranked.",
      "tab:perclass9", "@{}ll" + "r" * 11 + "@{}"))

# ------------------------------------------------------------------------- selected models
mo = pd.read_csv(D("models.csv"))
h2o_name = lambda mid: ("Stacked (all models)" if mid.startswith("StackedEnsemble_AllModels") else
                        "Stacked (best of family)" if mid.startswith("StackedEnsemble_BestOfFamily") else
                        "XGBoost" if mid.startswith("XGBoost") else "GBM" if mid.startswith("GBM") else
                        "Deep learning" if mid.startswith("DeepLearning") else mid.split("_")[0])
ask_name = {"LDA": "LDA", "LibLinear_SVC": "linear SVC", "AdaboostClassifier": "AdaBoost", "RandomForest": "random forest",
            "MLPClassifier": "MLP", "ExtraTreesClassifier": "extra trees", "KNearestNeighborsClassifier": "$k$-NN",
            "GradientBoostingClassifier": "gradient boosting"}


def ask_weights(sub):
    share = {}
    for s in sub.members.dropna():
        for item in s.split(";"):
            est, w = item.rsplit(":", 1)
            share[est] = share.get(est, 0.0) + float(w) / len(sub)
    tot = sum(share.values())
    return sorted(((ask_name.get(k, k), 100 * v / tot) for k, v in share.items()), key=lambda x: -x[1])


rows = ["Setup & H2O AutoML: leader type and number of seeds & Auto-sklearn: members, median (range) & Auto-sklearn: share of ensemble weight \\\\", "\\midrule"]
for cond, lab in SETUPS:
    ds, sp, td = cond.split("|")
    sub_h = mo[(mo.dataset == ds) & (mo.split == sp) & (mo.train_dist == td) & (mo.framework == "h2o")]
    sub_a = mo[(mo.dataset == ds) & (mo.split == sp) & (mo.train_dist == td) & (mo.framework == "autosklearn")]
    leaders = sub_h.leader_id.map(h2o_name).value_counts()
    hcell = ", ".join(f"{n}: {c}" for n, c in leaders.items())
    w = ask_weights(sub_a)[:3]
    acell = ", ".join(f"{n} {v:.0f}\\%" for n, v in w)
    n = sub_a.n_members.astype(int)
    rows.append(f"{lab} & {hcell} & {int(n.median())} ({n.min()}--{n.max()}) & {acell} \\\\")
    if cond == "mems|group|oversample":
        rows.append("\\midrule")
write("table_models.tex", wrap("\n".join(rows),
      "Models selected by the frameworks over five seeds. H2O AutoML: type of the leader model. Auto-sklearn: ensemble size and weight share per model type.",
      "tab:models", "@{}lp{1.8in}cp{1.9in}@{}"))

# per-seed selected models (appendix)
rows = ["Setup & Seed & H2O AutoML: leader & Auto-sklearn: members & Auto-sklearn: weight share, top three types \\\\", "\\midrule"]
for cond, lab in SETUPS:
    ds, sp, td = cond.split("|")
    for s in range(5):
        h = mo[(mo.dataset == ds) & (mo.split == sp) & (mo.train_dist == td) & (mo.framework == "h2o") & (mo.seed == s)]
        a = mo[(mo.dataset == ds) & (mo.split == sp) & (mo.train_dist == td) & (mo.framework == "autosklearn") & (mo.seed == s)]
        w = ask_weights(a)[:3]
        rows.append(f"{lab if s == 0 else ''} & {s} & \\texttt{{{esc(h.leader_id.iloc[0])}}} & {int(a.n_members.iloc[0])} & "
                    + ", ".join(f"{n} {v:.0f}\\%" for n, v in w) + " \\\\")
    if cond != SETUPS[-1][0]:
        rows.append("\\midrule")
write("table_models_seed.tex", wrap("\n".join(rows),
      "Selected models for every seed. H2O AutoML: identifier of the leaderboard leader. Auto-sklearn: number of ensemble members and weight share per model type.",
      "tab:modelsseed", "@{}llp{2.1in}cp{1.9in}@{}"))

# ------------------------------------------------------------------------- resources (pooled per dataset)
A = df[df.framework.isin(["autosklearn", "h2o"])].copy()
A["family"] = np.where(A.dataset == "mems", "MEMS", "Device 9")
A["mem_gb"] = A.peak_rss_mb / 1024
spec = [("wall_clock_fit_s", 0, True), ("mem_gb", 1, True), ("n_models_evaluated", 0, False), ("model_size_mb", 0, True),
        ("predict_latency_ms_per_sample", 2, True)]
rows = ["Dataset & Framework & Fit time (s) & Peak memory (GB) & Models evaluated & Model size (MB) & Latency (ms/sample) \\\\", "\\midrule"]
for fam in ["MEMS", "Device 9"]:
    sub = {fw: A[(A.family == fam) & (A.framework == fw)] for fw in ["autosklearn", "h2o"]}
    win = {c: (best_set({fw: round(sub[fw][c].median(), nd) for fw in sub}, "min") if rank else set()) for c, nd, rank in spec}
    for fw in ["autosklearn", "h2o"]:
        x = sub[fw]
        cs = [bold(f"{x[c].median():.{nd}f} ({x[c].min():.{nd}f}--{x[c].max():.{nd}f})", fw in win[c]) for c, nd, rank in spec]
        rows.append(f"{fam} & {NAME[fw]} & " + " & ".join(cs) + " \\\\")
    if fam == "MEMS":
        rows.append("\\midrule")
write("table_resources.tex", wrap("\n".join(rows),
      "Resource use: median (minimum--maximum) over runs. Lower is better. Bold: lowest median per dataset; models evaluated is not ranked.",
      "tab:resources", "@{}llrrrrr@{}"))

# resources per setup (appendix)
rows = ["Setup & Framework & Fit time (s) & Peak memory (GB) & Models evaluated & Model size (MB) & Latency (ms/sample) \\\\", "\\midrule"]
for cond, lab in SETUPS:
    group = ["autosklearn", "h2o"] + (["h2o-bal"] if cond == "mems|group|full" else [])
    win = {}
    for c, nd, rank in spec:
        vals_ = {}
        for fw in group:
            s = df[(df.cond == cond) & (df.framework == fw)].copy()
            s["mem_gb"] = s.peak_rss_mb / 1024
            vals_[fw] = round(s[c].mean(), nd)
        win[c] = best_set(vals_, "min") if rank else set()
    for i, fw in enumerate(group):
        s = df[(df.cond == cond) & (df.framework == fw)].copy()
        s["mem_gb"] = s.peak_rss_mb / 1024
        cs = [bold(f"{s[c].mean():.{nd}f} $\\pm$ {s[c].std(ddof=1):.{nd}f}", fw in win[c]) for c, nd, rank in spec]
        rows.append(f"{lab if i == 0 else ''} & {NAME[fw]} & " + " & ".join(cs) + " \\\\")
    if cond != SETUPS[-1][0]:
        rows.append("\\midrule")
write("table_resources_setup.tex", wrap("\n".join(rows),
      "Resource use per setup: mean $\\pm$ standard deviation over five seeds. Lower is better. Bold: lowest mean per setup; models evaluated is not ranked.",
      "tab:resourcessetup", "@{}llrrrrr@{}"))

# ------------------------------------------------------------------------- permutation importance
pf = pd.read_csv(D("pfi.csv"))
ma = pf[pf.framework == "autosklearn"].groupby("feature").importance.agg(["mean", "std"])
mh = pf[pf.framework == "h2o"].groupby("feature").importance.agg(["mean", "std"])
order = ((ma["mean"] + mh["mean"]) / 2).sort_values(ascending=False).index
ba, bh = ma["mean"][order].round(3).idxmax(), mh["mean"][order].round(3).idxmax()


def pfi_rows(features):
    out = []
    for f in features:
        a_mean, a_std = ma.loc[f, "mean"], ma.loc[f, "std"]
        h_mean, h_std = mh.loc[f, "mean"], mh.loc[f, "std"]
        out.append(f"\\texttt{{{esc(f)}}} & {bold(f'{a_mean:.3f} $\\pm$ {a_std:.3f}', f == ba)} & "
                   f"{bold(f'{h_mean:.3f} $\\pm$ {h_std:.3f}', f == bh)} \\\\")
    return out


write("table_pfi.tex", wrap("Feature & Auto-sklearn & H2O AutoML \\\\\n\\midrule\n" + "\n".join(pfi_rows(order[:8])),
      "Permutation importance (macro-F1 drop, mean $\\pm$ standard deviation over five seeds), device 9. Bold: largest value per column.",
      "tab:pfi", "@{}lrr@{}", star=False))
write("table_pfi_full.tex", wrap("Feature & Auto-sklearn & H2O AutoML \\\\\n\\midrule\n" + "\n".join(pfi_rows(order)),
      "Complete permutation importance of all 20 features of device 9 (macro-F1 drop, mean $\\pm$ standard deviation over five seeds).",
      "tab:pfifull", "@{}lrr@{}", star=False))

# ------------------------------------------------------------------------- paired differences
P = df.pivot_table(index=["cond", "seed"], columns="framework", values="macro_f1")
pairs = [("autosklearn", "h2o", "A$-$H"), ("autosklearn", "baseline-rf", "A$-$RF"), ("h2o", "baseline-rf", "H$-$RF"),
         ("autosklearn", "baseline-mlp", "A$-$MLP"), ("h2o", "baseline-mlp", "H$-$MLP"),
         ("autosklearn", "baseline-dt", "A$-$DT"), ("h2o", "baseline-dt", "H$-$DT")]
rows = ["Setup & " + " & ".join(p[2] for p in pairs) + " \\\\", "\\midrule"]
for cond, lab in SETUPS:
    cs = []
    for x, y, _ in pairs:
        dd = (P.loc[cond][x] - P.loc[cond][y]).dropna()
        wn, ls = int((dd > 0).sum()), int((dd < 0).sum())
        cs.append(bold(f"{dd.mean():+.3f} ({wn}/{ls})", wn == len(dd) and len(dd) > 0))
    rows.append(f"{lab} & " + " & ".join(cs) + " \\\\")
write("table_paired.tex", wrap("\n".join(rows),
      "Paired macro-F1 differences (first minus second method; seeds won/lost in parentheses). A: Auto-sklearn, H: H2O AutoML, DT: decision tree. Bold: first method won all five seeds.",
      "tab:paired", "@{}lrrrrrrr@{}"))

# ------------------------------------------------------------------------- summary with check marks
metrics = [("macro_f1", "Macro-F1"), ("balanced_accuracy", "Balanced accuracy"), ("accuracy", "Accuracy")]
auto = ["autosklearn", "h2o"]
allm = ["autosklearn", "h2o", "baseline-rf", "baseline-mlp", "baseline-dt"]
short = {"autosklearn": "Auto-sklearn", "h2o": "H2O AutoML", "baseline-rf": "Random forest", "baseline-mlp": "MLP", "baseline-dt": "Decision tree"}
rows = ["Setup & \\multicolumn{2}{c}{Macro-F1} & \\multicolumn{2}{c}{Balanced accuracy} & \\multicolumn{2}{c}{Accuracy} & Seeds won & Best of all methods \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        " & Auto-sklearn & H2O AutoML & Auto-sklearn & H2O AutoML & Auto-sklearn & H2O AutoML & (A/H) & (macro-F1) \\\\", "\\midrule"]
for cond, lab in SETUPS:
    cs = []
    for m, _ in metrics:
        means = {fw: round(df[(df.cond == cond) & (df.framework == fw)][m].mean(), 3) for fw in auto}
        top = max(means.values())
        for fw in auto:
            cs.append(f"{means[fw]:.3f}~" + ("\\cmark" if means[fw] == top else "\\phantom{\\cmark}"))
    dd = (P.loc[cond]["autosklearn"] - P.loc[cond]["h2o"]).dropna()
    allmeans = {fw: round(df[(df.cond == cond) & (df.framework == fw)]["macro_f1"].mean(), 3) for fw in allm}
    topall = max(allmeans.values())
    names = " / ".join(short[fw] for fw in allm if allmeans[fw] == topall)
    rows.append(f"{lab} & " + " & ".join(cs) + f" & {int((dd > 0).sum())}/{int((dd < 0).sum())} & {names} \\\\")
    if cond == "mems|group|oversample":
        rows.append("\\midrule")
write("table_summary.tex", wrap("\n".join(rows),
      "Summary per setup. \\cmark: better of the two AutoML frameworks (ties mark both). Last column: best of all methods by macro-F1.",
      "tab:summary", "@{}lcccccccl@{}"))

# ------------------------------------------------------------------------- appendix: complete metrics
full_metrics = [("accuracy", "Accuracy"), ("balanced_accuracy", "Bal. acc."), ("macro_precision", "Macro-P"),
                ("macro_recall", "Macro-R"), ("macro_f1", "Macro-F1"), ("weighted_f1", "Weighted F1"), ("mcc", "MCC")]


def full_table(setups, name, caption, label):
    rows = ["Method & " + " & ".join(m[1] for m in full_metrics) + " \\\\", "\\midrule"]
    for k, (cond, lab) in enumerate(setups):
        rows.append(f"\\multicolumn{{{len(full_metrics) + 1}}}{{@{{}}l}}{{\\textit{{{lab}}}}} \\\\")
        present = [fw for fw in ORDER if len(sel(cond, fw, "macro_f1")) > 0]
        vv = {(fw, m): ms(cond, fw, m) for fw in present for m, _ in full_metrics}
        for fw in present:
            cs = []
            for m, _ in full_metrics:
                win = best_set({f: round(vv[(f, m)][0], 3) for f in present if f != "baseline-dummy"})
                cs.append(cell(vv[(fw, m)], 3, fw in win))
            rows.append(f"{NAME[fw]} & " + " & ".join(cs) + " \\\\")
        if k < len(setups) - 1:
            rows.append("\\midrule")
    write(name, wrap("\n".join(rows), caption, label, "@{}l" + "r" * len(full_metrics) + "@{}", tabcolsep=2))


full_table(SETUPS[:3], "table_full_mems.tex",
           "Complete results on MEMS: mean $\\pm$ standard deviation over five seeds. Bold: highest mean per column and setup.", "tab:fullmems")
full_table(SETUPS[3:], "table_full_dev9.tex",
           "Complete results on device 9: mean $\\pm$ standard deviation over five seeds. Bold: highest mean per column and setup.", "tab:fulldev9")

# per-class recall, MEMS (appendix)
rows = ["Method & c1 & c2 & c3 \\\\", "\\midrule"]
for k, (cond, lab) in enumerate(SETUPS[:3]):
    rows.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{lab}}}}} \\\\")
    present = [fw for fw in ORDER if len(sel(cond, fw, "macro_f1")) > 0]
    for fw in present:
        cs = []
        for c in ("recall::class_1", "recall::class_2", "recall::class_3"):
            win = best_set({f: round(sel(cond, f, c).mean(), 2) for f in present if f != "baseline-dummy"})
            cs.append(cell(ms(cond, fw, c), 2, fw in win))
        rows.append(f"{NAME[fw]} & " + " & ".join(cs) + " \\\\")
    if k < 2:
        rows.append("\\midrule")
write("table_perclass_mems.tex", wrap("\n".join(rows),
      "MEMS: per-class recall (c1 normal, c2 near-failure, c3 failure), mean $\\pm$ standard deviation over five seeds. Bold: highest mean per class and setup.",
      "tab:perclassmems", "@{}lrrr@{}", star=False, tabcolsep=3))

# per-class recall, device 9, all methods (appendix)
rows = ["Variant & Method & " + " & ".join(f"c{i}" for i in range(1, 12)) + " \\\\", "\\midrule"]
for cond, lab, amb in [(d, "Deduplicated", set()), (r_, "Raw", {1, 3, 4})]:
    present = ["autosklearn", "h2o", "baseline-rf", "baseline-mlp", "baseline-dt", "baseline-dummy"]
    means = {fw: df[(df.cond == cond) & (df.framework == fw)][cls].mean().values for fw in present}
    wins = [best_set({fw: round(means[fw][i], 2) for fw in present if fw != "baseline-dummy"}) if (i + 1) not in amb else set() for i in range(11)]
    for fw in present:
        rows.append(f"{lab if fw == present[0] else ''} & {NAME[fw]} & " + " & ".join(bold(f"{means[fw][i]:.2f}", fw in wins[i]) for i in range(11)) + " \\\\")
    rows.append("\\midrule")
rows.pop()
write("table_perclass9_all.tex", wrap("\n".join(rows),
      "Device 9: mean per-class recall over five seeds for all methods. Bold: best method per class and variant; raw c1, c3, c4 are not ranked.",
      "tab:perclass9all", "@{}ll" + "r" * 11 + "@{}"))

# per-seed macro-F1 (appendix)
rows = ["Method & Seed 0 & Seed 1 & Seed 2 & Seed 3 & Seed 4 & Mean \\\\", "\\midrule"]
for k, (cond, lab) in enumerate(SETUPS):
    rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{lab}}}}} \\\\")
    present = [fw for fw in ORDER if len(sel(cond, fw, "macro_f1")) > 0]
    for fw in present:
        s = df[(df.cond == cond) & (df.framework == fw)].set_index("seed").macro_f1
        rows.append(f"{NAME[fw]} & " + " & ".join(f"{s.loc[i]:.3f}" for i in range(5)) + f" & {s.mean():.3f} \\\\")
    if k < len(SETUPS) - 1:
        rows.append("\\midrule")
write("table_perseed.tex", wrap("\n".join(rows),
      "Macro-F1 of every run. The seed determines the split, so rows within a setup are paired by seed.",
      "tab:perseed", "@{}lrrrrrr@{}", star=False, tabcolsep=3))

print("tables written:", sorted(f for f in os.listdir(os.path.join(ROOT, "tables")) if f.endswith(".tex")))
