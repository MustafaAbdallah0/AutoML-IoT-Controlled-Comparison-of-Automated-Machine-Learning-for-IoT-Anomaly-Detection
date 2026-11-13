"""run_baselines.py - non-AutoML reference models on the identical split (default hyper-parameters)."""
import argparse, json, os, time
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from env_info import collect_env
from iot_data import make_split, DEFAULT_DATA_DIR
from metrics_utils import compute_metrics


def models(seed):
    return {"baseline-dummy": DummyClassifier(strategy="most_frequent"),
            "baseline-dt": DecisionTreeClassifier(random_state=seed),
            "baseline-rf": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1),
            "baseline-mlp": make_pipeline(StandardScaler(), MLPClassifier(max_iter=200, random_state=seed))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--split", choices=["random", "group"], default="random")
    p.add_argument("--train_dist", default="full")
    p.add_argument("--test_size", type=float, default=0.3)
    p.add_argument("--no_dedup", action="store_true")
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="results")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    split = make_split(a.dataset, a.split, a.seed, a.test_size, a.train_dist, a.data_dir,
                       dedup=False if a.no_dedup else None, max_rows=a.max_rows)
    for name, model in models(a.seed).items():
        ds_tag = a.dataset.replace(':', '') + ("-raw" if a.no_dedup and a.dataset.startswith("nbaiot") else "")
        tag = f"{ds_tag}__{a.split}__{a.train_dist.replace(':', '')}__{name}"
        t0 = time.time()
        model.fit(split.X_train.values, split.y_train.values)
        fit_s = time.time() - t0
        t1 = time.time()
        pred = model.predict(split.X_test.values)
        lat = 1000 * (time.time() - t1) / len(split.X_test)
        m, pc, cm = compute_metrics(split.y_test, pred, split.classes)
        res = {"tag": tag, "framework": name, "dataset": a.dataset, "split": a.split,
               "train_dist": a.train_dist, "seed": a.seed, "split_info": split.info, "metrics": m,
               "per_class": pc, "labels": split.classes, "confusion_matrix": cm,
               "resources": {"wall_clock_fit_s": round(fit_s, 1),
                             "predict_latency_ms_per_sample": round(lat, 4)}, "env": collect_env()}
        with open(os.path.join(a.out_dir, f"{tag}__s{a.seed}.json"), "w") as f:
            json.dump(res, f, indent=1)
        print(f"{name:15s} macro-F1={m['macro_f1']:.4f} acc={m['accuracy']:.4f}")


if __name__ == "__main__":
    main()
