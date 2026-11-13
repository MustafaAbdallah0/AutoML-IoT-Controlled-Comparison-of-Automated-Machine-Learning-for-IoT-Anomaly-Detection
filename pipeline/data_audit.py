"""data_audit.py - dataset quality facts for the paper (duplication, effective class sizes, runs)."""
import argparse, os
import pandas as pd
from iot_data import audit_dataset, DEFAULT_DATA_DIR

p = argparse.ArgumentParser()
p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
p.add_argument("--out", default="tables")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
rows = [audit_dataset(n, a.data_dir) for n in ["mems"] + [f"nbaiot:{k}" for k in range(1, 10)]]
df = pd.DataFrame(rows)
df.to_csv(os.path.join(a.out, "data_audit.csv"), index=False)
print(df.to_string(index=False))
nb = [r for r in rows if r["dataset"].startswith("nbaiot")]
min_dup = min(r["duplicate_pct"] for r in nb)
max_minc = max(r["min_unique_per_class"] for r in nb)
B = lambda t, on: f"\\textbf{{{t}}}" if on else t
tex = ["\\begin{table*}[t]", "\\centering",
       "\\caption{Audit of the released datasets. Cleaning removes exact duplicates and ambiguous vectors. \\textbf{Bold} (N-BaIoT): lowest duplicate share and largest smallest class.}",
       "\\label{tab:audit}", "\\scriptsize", "\\begin{tabular}{@{}lrrrrrp{1.5in}@{}}", "\\toprule",
       "Dataset & Classes & Rows & Unique rows & Dup.\\,\\% & Min unique/class & Classes with $<$50 unique rows \\\\", "\\midrule"]
for r in rows:
    nbr = r["dataset"].startswith("nbaiot")
    small = (r["classes_lt50_unique"] or "--").replace("_", " ").replace(",", ", ")
    tex.append(f"{r['dataset']} & {r['classes']} & {r['rows_raw']:,} & {r['rows_unique']:,} & "
               f"{B(str(r['duplicate_pct']), nbr and r['duplicate_pct'] == min_dup)} & "
               f"{B(format(r['min_unique_per_class'], ','), nbr and r['min_unique_per_class'] == max_minc)} & {small} " + "\\\\")
tex += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
open(os.path.join(a.out, "table_audit.tex"), "w").write("\n".join(tex))

# ---- label ambiguity (identical feature vectors under different labels) ----
from iot_data import audit_ambiguity
summ, per9 = [], None
for k in range(1, 10):
    s, pc = audit_ambiguity(f"nbaiot:{k}", a.data_dir)
    summ.append(s)
    if k == 9:
        per9 = pc
pd.DataFrame(summ).to_csv(os.path.join(a.out, "ambiguity_summary.csv"), index=False)
per9.to_csv(os.path.join(a.out, "ambiguity_device9.csv"), index=False)
print(pd.DataFrame(summ).to_string(index=False)); print(per9.to_string(index=False))
