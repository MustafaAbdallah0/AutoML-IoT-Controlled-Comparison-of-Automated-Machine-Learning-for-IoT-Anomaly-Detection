"""
run_automl.py - one controlled AutoML run (framework x dataset x split x training
distribution x seed). Both frameworks share data, split, seed, wall-clock budget,
thread count, metrics, resource monitor and result format.

Examples
  python run_automl.py --framework h2o         --dataset mems --split group --budget 1800 --seed 0
  python run_automl.py --framework autosklearn --dataset nbaiot:7 --budget 1800 --seed 0 --threads 8
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np

from env_info import collect_env
from iot_data import make_split, DEFAULT_DATA_DIR
from metrics_utils import compute_metrics, permutation_importance
from resource_monitor import ResourceMonitor


def acquire_lock(path):
    """One process per result cell. A lock left by a dead process is taken over."""
    import psutil
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                pid = int(open(path).read().strip())
            except Exception:
                pid = -1
            if pid > 0 and psutil.pid_exists(pid):
                return False
            try:
                os.remove(path)
            except OSError:
                pass
    return False


def free_port():
    """A random port p with p and p+1 free (H2O uses both)."""
    import random
    import socket
    for _ in range(100):
        p = random.randint(20000, 60000)
        socks = []
        try:
            for q in (p, p + 1):
                sk = socket.socket()
                sk.bind(("127.0.0.1", q))
                socks.append(sk)
            return p
        except OSError:
            continue
        finally:
            for sk in socks:
                sk.close()
    raise RuntimeError("no free port found for H2O")


class Fitted:
    def __init__(self, predict, info, n_evaluated, size_bytes, cleanup=None):
        self.predict, self.info, self.n_evaluated = predict, info, n_evaluated
        self.size_bytes, self.cleanup = size_bytes, cleanup


def fit_h2o(split, a, workdir):
    import h2o
    from h2o.automl import H2OAutoML
    port = free_port()                      # own cluster per run: concurrent runs must not share one JVM
    h2o.init(ip="127.0.0.1", port=port, name=f"iot_{os.getpid()}", max_mem_size=a.h2o_mem,
             nthreads=a.threads, verbose=False, strict_version_check=False)
    h2o.no_progress()
    train = split.X_train.copy()
    train["label"] = split.y_train.values
    htr = h2o.H2OFrame(train)
    htr["label"] = htr["label"].asfactor()
    kw = dict(max_runtime_secs=a.budget, seed=a.seed, sort_metric="mean_per_class_error",
              balance_classes=a.h2o_balance_classes)
    if a.h2o_max_models:
        kw["max_models"] = a.h2o_max_models
    if a.h2o_exclude:
        kw["exclude_algos"] = a.h2o_exclude.split(",")
    aml = H2OAutoML(**kw)
    aml.train(x=list(split.X_train.columns), y="label", training_frame=htr)
    lb = aml.leaderboard.as_data_frame()
    leader = aml.leader
    path = h2o.save_model(leader, path=workdir, force=True)

    def predict(X):
        return leader.predict(h2o.H2OFrame(X)).as_data_frame()["predict"].astype(str).values

    info = {"leader": leader.model_id, "leader_algo": leader.algo,
            "leaderboard_top5": lb["model_id"].head(5).tolist(), "balance_classes": a.h2o_balance_classes}
    return Fitted(predict, info, len(lb), os.path.getsize(path),
                  cleanup=lambda: h2o.cluster().shutdown(prompt=False))


def fit_autosklearn(split, a, workdir):
    import joblib
    import autosklearn.classification as ask
    import autosklearn.metrics as askm
    tmp = os.path.join(workdir, "ask_tmp")
    clf = ask.AutoSklearnClassifier(
        time_left_for_this_task=a.budget, per_run_time_limit=a.per_run or max(30, a.budget // 15),
        metric=askm.f1_macro, seed=a.seed, n_jobs=a.threads, memory_limit=a.ask_mem_mb,
        resampling_strategy="holdout", resampling_strategy_arguments={"train_size": 0.67},
        ensemble_kwargs={"ensemble_size": 50}, tmp_folder=tmp, delete_tmp_folder_after_terminate=False)
    Xtr = split.X_train.values.astype(np.float64)
    clf.fit(Xtr, split.y_train.values)
    if a.ask_refit:
        clf.refit(Xtr, split.y_train.values)
    models = clf.get_models_with_weights()
    members = []
    for w, m in models:
        est = m.steps[-1][1] if hasattr(m, "steps") else m
        name = type(getattr(est, "choice", est)).__name__
        members.append({"weight": round(float(w), 4), "estimator": name})
    pkl = os.path.join(workdir, "ask_model.pkl")
    joblib.dump(clf, pkl)
    n_eval = len(clf.automl_.runhistory_.data)
    cols = list(split.X_train.columns)

    def predict(X):
        return clf.predict(X[cols].values.astype(np.float64)).astype(str)

    info = {"n_ensemble_members": len(models), "members": members, "refit": a.ask_refit,
            "statistics": clf.sprint_statistics().replace("\n", " | ")}
    return Fitted(predict, info, n_eval, os.path.getsize(pkl),
                  cleanup=lambda: shutil.rmtree(tmp, ignore_errors=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--framework", choices=["h2o", "autosklearn"], required=True)
    p.add_argument("--dataset", required=True, help="mems | nbaiot:<1-9>")
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--split", choices=["random", "group"], default="random")
    p.add_argument("--train_dist", default="full", help="full | cap:N | oversample")
    p.add_argument("--test_size", type=float, default=0.3)
    p.add_argument("--no_dedup", action="store_true", help="keep exact duplicates (nbaiot)")
    p.add_argument("--max_rows", type=int, default=None, help="subsample (smoke tests only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget", type=int, default=1800, help="wall-clock seconds, both frameworks")
    p.add_argument("--threads", type=int, default=1, help="H2O nthreads AND Auto-sklearn n_jobs")
    p.add_argument("--per_run", type=int, default=None, help="Auto-sklearn per-config limit (s)")
    p.add_argument("--ask_mem_mb", type=int, default=3072)
    p.add_argument("--ask_refit", action="store_true")
    p.add_argument("--h2o_mem", default="4G")
    p.add_argument("--h2o_balance_classes", action="store_true")
    p.add_argument("--h2o_max_models", type=int, default=None)
    p.add_argument("--h2o_exclude", default=None, help="comma list, e.g. DeepLearning,XGBoost")
    p.add_argument("--pfi", action="store_true", help="permutation importance of the final model")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    ds_tag = a.dataset.replace(':', '') + ("-raw" if a.no_dedup and a.dataset.startswith("nbaiot") else "")
    tag = f"{ds_tag}__{a.split}__{a.train_dist.replace(':', '')}__{a.framework}"
    if a.framework == "h2o" and a.h2o_balance_classes:
        tag += "-bal"
    out = os.path.join(a.out_dir, f"{tag}__s{a.seed}.json")
    os.makedirs(a.out_dir, exist_ok=True)
    if os.path.exists(out) and not a.overwrite:
        print("exists, skipping:", out)
        return
    if not acquire_lock(out + ".lock"):
        print("another process is already running this cell, skipping:", out)
        return

    split = make_split(a.dataset, a.split, a.seed, a.test_size, a.train_dist, a.data_dir,
                       dedup=False if a.no_dedup else None, max_rows=a.max_rows)
    print(f"[{tag} s{a.seed}] train={len(split.y_train)} test={len(split.y_test)} "
          f"classes={len(split.classes)} budget={a.budget}s threads={a.threads}", flush=True)

    workdir = tempfile.mkdtemp(prefix="iotautoml_")
    mon = ResourceMonitor(interval=0.5)
    mon.start()                                   # started first: captures the H2O JVM child too
    t0 = time.time()
    try:
        fitted = (fit_h2o if a.framework == "h2o" else fit_autosklearn)(split, a, workdir)
        fit_s = time.time() - t0
        mon.stop()
        res = mon.summary()
        sub = split.X_test.iloc[:5000]
        t1 = time.time()
        fitted.predict(sub)
        latency_ms = 1000 * (time.time() - t1) / len(sub)
        y_pred = fitted.predict(split.X_test)
        m, per_class, cm = compute_metrics(split.y_test, y_pred, split.classes)
        pfi = (permutation_importance(fitted.predict, split.X_test, split.y_test, split.classes,
                                      seed=a.seed) if a.pfi else None)
        result = {
            "tag": tag, "framework": a.framework, "dataset": a.dataset, "split": a.split,
            "train_dist": a.train_dist, "seed": a.seed, "config": vars(a),
            "split_info": split.info, "metrics": m, "per_class": per_class,
            "labels": split.classes, "confusion_matrix": cm,
            "resources": {"wall_clock_fit_s": round(fit_s, 1), "peak_rss_mb": res["peak_rss_mb"],
                          "peak_cpu_percent": res["peak_cpu_percent"],
                          "mean_cpu_percent": res["mean_cpu_percent"],
                          "n_models_evaluated": fitted.n_evaluated,
                          "model_size_mb": round(fitted.size_bytes / 1024 ** 2, 2),
                          "predict_latency_ms_per_sample": round(latency_ms, 4)},
            "model_info": fitted.info, "pfi": pfi, "env": collect_env(),
        }
        with open(out, "w") as f:
            json.dump(result, f, indent=1)
        print(f"macro-F1={m['macro_f1']:.4f} acc={m['accuracy']:.4f} bal-acc={m['balanced_accuracy']:.4f} "
              f"fit={fit_s:.0f}s peakRSS={res['peak_rss_mb']:.0f}MB -> {out}")
        if fitted.cleanup:
            fitted.cleanup()
    except Exception:
        mon.stop()
        traceback.print_exc()
        sys.exit(1)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        try:
            os.remove(out + ".lock")
        except OSError:
            pass


if __name__ == "__main__":
    main()
