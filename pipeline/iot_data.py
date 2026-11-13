"""
iot_data.py
Loading, auditing and leakage-aware splitting for the two IoT datasets of the
XAI-IoT repository (agummadi1/XAI_for_IoT_Systems):

    mems         MEMS vibration data (x, y, z -> Normal / Near-failure / Failure)
    nbaiot:<k>   N-BaIoT device k = 1..9 (top-20 features, benign + botnet classes)

Design points (each one answers a concern raised for the earlier AutoML paper):
  * both AutoML frameworks receive EXACTLY the same train/test arrays;
  * N-BaIoT files contain heavy exact duplication (see data_audit.py), so exact
    duplicates and label-conflicting rows are removed before splitting;
  * MEMS is a time series stored in long contiguous blocks; a `group` split holds
    out whole blocks ("runs") to avoid temporal leakage;
  * the training distribution (full / capped / oversampled) is a separate factor,
    applied identically to every framework.
"""
import itertools
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = os.environ.get("IOT_DATA_DIR", "XAI_for_IoT_Systems/Datasets")
MEMS_FILE = "mems_dataset.csv"
NBAIOT_DIR = "IOT Top 20 features datasets"

# Label ids -> names. MEMS names follow the ALE figure captions of the XAI-IoT paper
# (1 normal, 2 near-failure, 3 failure); N-BaIoT names follow its class list.
MEMS_CLASSES = {1: "Normal", 2: "Near-failure", 3: "Failure"}
NBAIOT_CLASSES = {1: "benign", 2: "gafgyt.combo", 3: "gafgyt.junk", 4: "gafgyt.scan",
                  5: "gafgyt.tcp", 6: "gafgyt.udp", 7: "mirai.ack", 8: "mirai.scan",
                  9: "mirai.syn", 10: "mirai.udp", 11: "mirai.udpplain"}


def _names(kind, assumed):
    """Class names. The CSVs hold integer ids only; the id->name mapping below is taken from the
    XAI-IoT paper text and is UNVERIFIED, so neutral ids are used unless IOT_USE_NAMES=1."""
    if os.environ.get("IOT_USE_NAMES") == "1":
        return assumed
    return {k: f"class_{k}" for k in assumed}


def _clean(c):
    return re.sub(r"[^0-9A-Za-z_]", "_", str(c))


def parse_name(name):
    if name == "mems":
        return "mems", None
    m = re.fullmatch(r"nbaiot[:_]?([1-9])", name)
    if m:
        return "nbaiot", int(m.group(1))
    raise ValueError(f"Unknown dataset '{name}'. Use 'mems' or 'nbaiot:<1-9>'.")


def label_runs(labels):
    """Run id of each row = index of its contiguous block of identical labels."""
    ids = []
    for i, (_, g) in enumerate(itertools.groupby(labels)):
        ids.extend([i] * len(list(g)))
    return np.asarray(ids)


def _dedup(df):
    feats = [c for c in df.columns if c not in ("label", "run")]
    n0 = len(df)
    d1 = df.drop_duplicates(subset=feats + ["label"])
    conflict = d1.duplicated(subset=feats, keep=False)   # same features, different labels
    d2 = d1[~conflict]
    return d2.reset_index(drop=True), {"rows_raw": n0, "rows_after_exact_dedup": len(d1),
                                       "conflict_rows_removed": int(conflict.sum()),
                                       "rows_final": len(d2)}


def load_dataset(name, data_dir=DEFAULT_DATA_DIR, dedup=None):
    """Return (df, class_names, meta). df has feature columns, int `label`, int `run`."""
    kind, dev = parse_name(name)
    if kind == "mems":
        df = pd.read_csv(os.path.join(data_dir, MEMS_FILE))[["x", "y", "z", "label"]].copy()
        df["run"] = label_runs(df["label"].values)   # inferred: file has no session/RPM column
        names = _names("mems", MEMS_CLASSES)
        do_dedup = False if dedup is None else dedup
    else:
        path = os.path.join(data_dir, NBAIOT_DIR, f"device{dev}_top_20_features.csv")
        df = pd.read_csv(path)
        df.columns = [_clean(c) for c in df.columns]
        df["run"] = -1
        names = _names("nbaiot", NBAIOT_CLASSES)
        do_dedup = True if dedup is None else dedup
    meta = {"dataset": name, "kind": kind, "device": dev, "dedup": do_dedup}
    if do_dedup:
        df, st = _dedup(df)
        meta.update(st)
    else:
        meta.update({"rows_raw": len(df), "rows_final": len(df)})
    return df, names, meta


@dataclass
class Split:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    classes: list
    info: dict = field(default_factory=dict)


def _random_idx(y, seed, test_size):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    return tr, te


def _grouped_idx(y, runs, seed, test_size):
    """Class-aware hold-out of whole runs: ~test_size of each class's runs go to test
    (at least one, never all), so every class is present in both partitions."""
    rng = np.random.RandomState(seed)
    test_runs = []
    for c in np.unique(y):
        rc = np.unique(runs[y == c])
        if len(rc) < 2:
            raise ValueError(f"class {c} has a single run; cannot hold out a whole run")
        k = min(len(rc) - 1, max(1, int(round(test_size * len(rc)))))
        test_runs.extend(rng.choice(rc, size=k, replace=False).tolist())
    te_mask = np.isin(runs, test_runs)
    return np.where(~te_mask)[0], np.where(te_mask)[0]


def _train_dist(Xtr, ytr, spec, seed):
    rng = np.random.RandomState(seed)
    if spec == "full":
        return Xtr, ytr
    if spec.startswith("cap:"):
        cap = int(spec.split(":")[1])
        keep = np.concatenate([rng.permutation(np.where(ytr.values == c)[0])[:cap]
                               for c in ytr.unique()])
    elif spec == "oversample":
        target = int(ytr.value_counts().max())
        keep = np.concatenate([rng.choice(np.where(ytr.values == c)[0], size=target, replace=True)
                               for c in ytr.unique()])
    else:
        raise ValueError(f"unknown train_dist '{spec}' (full | cap:N | oversample)")
    keep = rng.permutation(keep)
    return Xtr.iloc[keep], ytr.iloc[keep]


def make_split(name, split="random", seed=0, test_size=0.3, train_dist="full",
               data_dir=DEFAULT_DATA_DIR, dedup=None, max_rows=None):
    df, names, meta = load_dataset(name, data_dir, dedup)
    if max_rows and len(df) > max_rows:        # stratified subsample, for smoke tests only
        df = (df.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(max(2, int(round(max_rows * len(g) / len(df)))),
                                          random_state=seed))
                .reset_index(drop=True))
    feats = [c for c in df.columns if c not in ("label", "run")]
    y_int = df["label"].values
    if split == "random":
        tr, te = _random_idx(y_int, seed, test_size)
    elif split == "group":
        if meta["kind"] != "mems":
            raise ValueError("group split needs run ids, available for 'mems' only")
        tr, te = _grouped_idx(y_int, df["run"].values, seed, test_size)
    else:
        raise ValueError("split must be 'random' or 'group'")
    y = df["label"].map(lambda k: names.get(int(k), f"class_{int(k)}"))
    Xtr, ytr = df.iloc[tr][feats], y.iloc[tr]
    Xte, yte = df.iloc[te][feats], y.iloc[te]
    n_train_before = len(ytr)
    Xtr, ytr = _train_dist(Xtr, ytr, train_dist, seed)
    classes = sorted(y.unique())
    info = dict(meta, split=split, seed=seed, test_size=test_size, train_dist=train_dist,
                n_features=len(feats), n_train_before_dist=n_train_before, n_train=len(ytr),
                n_test=len(yte), train_counts=ytr.value_counts().to_dict(),
                test_counts=yte.value_counts().to_dict())
    return Split(Xtr.reset_index(drop=True), ytr.reset_index(drop=True),
                 Xte.reset_index(drop=True), yte.reset_index(drop=True), classes, info)


def audit_dataset(name, data_dir=DEFAULT_DATA_DIR):
    """Data-quality facts reported in the paper (duplication, effective class sizes, runs)."""
    kind, dev = parse_name(name)
    raw, names, _ = load_dataset(name, data_dir, dedup=False)
    clean, _, meta = load_dataset(name, data_dir, dedup=True)
    uniq = clean["label"].value_counts()
    out = {"dataset": name, "classes": int(raw["label"].nunique()), "rows_raw": len(raw),
           "rows_unique": int(meta["rows_final"]),
           "duplicate_pct": round(100 * (1 - meta["rows_final"] / len(raw)), 1),
           "min_unique_per_class": int(uniq.min()),
           "classes_lt50_unique": ",".join(names.get(int(k), str(k)) for k in uniq[uniq < 50].index)}
    if kind == "mems":
        runs = label_runs(raw["label"].values)
        out.update({"runs": int(runs.max() + 1),
                    "median_run_len": int(pd.Series(runs).value_counts().median())})
    return out


def audit_ambiguity(name, data_dir=DEFAULT_DATA_DIR):
    """Label ambiguity of the (raw) data: feature vectors that occur under more than one label.
    Returns (summary dict, per-class DataFrame). `ceiling_*` = recall obtained by labelling every
    ambiguous vector with its majority label and every other vector correctly (an upper bound for
    any classifier that cannot tell identical inputs apart; computed on the full raw file)."""
    df, names, _ = load_dataset(name, data_dir, dedup=False)
    feats = [c for c in df.columns if c not in ("label", "run")]
    d = df[["label"]].copy()
    d["vec"] = pd.util.hash_pandas_object(df[feats], index=False).values
    n_lab = d.groupby("vec")["label"].nunique()
    d["ambiguous"] = d["vec"].map(n_lab) > 1
    cnt = d.groupby(["vec", "label"]).size().rename("n").reset_index()
    best = cnt.loc[cnt.groupby("vec")["n"].idxmax()][["vec", "label"]].rename(columns={"label": "best"})
    d = d.merge(best, on="vec")
    d["hit"] = d["label"] == d["best"]
    rows = []
    for c, g in d.groupby("label"):
        uv = g["vec"].nunique()
        rows.append({"class": names.get(int(c), str(c)), "raw_rows": len(g), "unique_vectors": uv,
                     "duplicate_pct": round(100 * (1 - uv / len(g)), 1),
                     "rows_in_ambiguous_vectors_pct": round(100 * g["ambiguous"].mean(), 1),
                     "ceiling_recall": round(float(g["hit"].mean()), 3)})
    per_class = pd.DataFrame(rows)
    summ = {"dataset": name, "rows_raw": len(d), "ambiguous_vectors": int((n_lab > 1).sum()),
            "rows_in_ambiguous_vectors_pct": round(100 * d["ambiguous"].mean(), 1),
            "ceiling_accuracy": round(float(d["hit"].mean()), 3),
            "ceiling_balanced_accuracy": round(float(per_class["ceiling_recall"].mean()), 3)}
    return summ, per_class
