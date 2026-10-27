"""Turn the three voters into congestion levels: choose and fit on the practice labels, test once.

    python -m src.congestion.ensemble_eval fit      # practice labels only -> ensemble_config.json
    python -m src.congestion.ensemble_eval test     # once: the frozen config on the test labels
    python -m src.congestion.ensemble_eval apply    # every frame -> congestion_levels.csv

Inputs (results/congestion/): voter_occupancy.csv (A), voter_background.csv (B), the CLIP crop trials
in clip_trials/voter_clip_<crop>.csv (C, scored on the labelled frames only), labeling/tasks.json and
labeling/congestion_labels.json (the human labels: free / moderate / congested).

Road-level scores. The labels follow the more congested road when a camera's zones cover two roads,
so every voter has two versions: "union" (all zones together, as the voter scripts write it) and
"max" (the most congested zone). Which version each voter uses is chosen on the practice labels.

fit (practice labels only; the test labels are not read):
1. For each voter, choose the feature (A: vehicle count, PCU-weighted count or box occupancy; each as
   union or max; B: union or max; C: crop context / mask / zones, union or max) by 5-fold
   cross-validation repeated 20 times, scoring with linearly weighted Cohen's kappa.
2. Candidate methods, all cross-validated the same way:
   - A, B, C alone: two cut points on the feature (free | moderate | congested), set to maximise accuracy;
   - vote: the median of the three levels from A, B and C;
   - stack: multinomial logistic regression on log(1 + A count), A occupancy, B and C (standardised).
3. The primary method is the one with the highest cross-validated kappa; within 0.005 of the best,
   the simpler one wins (order A, B, C, vote, stack). Every candidate is then fitted on all
   practice labels and frozen in ensemble_config.json, together with a hash of the practice labels.

test (once; refuses to overwrite an existing result without --force):
- every frozen candidate on the test labels: accuracy, weighted kappa, macro F1, and congested-vs-rest
  precision / recall / F1, with 95% bootstrap intervals (2,000 resamples of test frames);
- the difference primary - A alone, with its bootstrap interval;
- per camera and day / night accuracy;
- population-weighted accuracy: the test sample over-represents peak hours, so each frame is also
  weighted by (frames of its camera and time stratum) / (test frames drawn from that stratum).

apply: the primary method (and every candidate whose inputs exist) on all frames. If the primary uses
C, voter_clip.csv must come from a full run with the chosen crop; the script prints the command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = ["free", "moderate", "congested"]
ORDER = ["A", "B", "C", "vote", "stack"]
REPS, FOLDS, GRID = 20, 5, 81


# ---------- data ----------

def zone_max(df: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [c for c in df.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]
    return df[cols].max(axis=1, skipna=True)


def load_a_b(root: Path) -> pd.DataFrame:
    a = pd.read_csv(root / "voter_occupancy.csv", dtype={"camera_id": str})
    b = pd.read_csv(root / "voter_background.csv", dtype={"camera_id": str})
    f = a[["image", "camera_id", "captured_at_sgt", "weekday", "hour", "is_day"]].copy()
    f["A_count_union"], f["A_count_max"] = a["n_vehicles"], zone_max(a, "n_z")
    if "pcu" in a:
        f["A_pcu_union"], f["A_pcu_max"] = a["pcu"], zone_max(a, "pcu_z")
    f["A_occ_union"], f["A_occ_max"] = a["occupancy"], zone_max(a, "occ_z")
    b = b.set_index("image")
    f["B_union"] = f["image"].map(b["bg_score"])
    f["B_max"] = f["image"].map(zone_max(b, "bg_z"))
    return f


def load_c(path: Path, crop: str) -> pd.DataFrame:
    c = pd.read_csv(path)
    out = pd.DataFrame({"image": c["image"], f"C_{crop}_union": c["clip_score"]})
    out[f"C_{crop}_max"] = c["clip_max"] if "clip_max" in c else c["clip_score"]
    return out


def _load_clip_parts(directory: Path, run_info: Path) -> list[pd.DataFrame]:
    parts = [load_c(p, p.stem.removeprefix("voter_clip_")) for p in sorted(directory.glob("voter_clip_*.csv"))]
    # A single --crop run writes voter_clip.csv without a crop suffix. Read the
    # crop name from clip_run.json so the default single-crop pipeline works too.
    plain = directory / "voter_clip.csv"
    if plain.is_file():
        crop = "context"
        if run_info.is_file():
            crop = json.loads(run_info.read_text(encoding="utf-8")).get("crop_mode", crop)
        parts.append(load_c(plain, crop))
    return parts


def load_trials(root: Path) -> pd.DataFrame | None:
    # Prefer the labeled-frame trials in clip_trials/. Fall back to the root
    # output of a default full clip_voter run, which is also what apply uses.
    parts = _load_clip_parts(root / "clip_trials", root / "clip_trials" / "clip_run.json")
    if not parts:
        parts = _load_clip_parts(root, root / "clip_run.json")
    if not parts:
        return None
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="image", how="outer")
    return out


def load_labels(root: Path, split: str) -> tuple[pd.DataFrame, str, dict]:
    labels = json.loads((root / "labeling" / "congestion_labels.json").read_text(encoding="utf-8"))
    tasks = json.loads((root / "labeling" / "tasks.json").read_text(encoding="utf-8"))
    if labels["task_hash"] != tasks["task_hash"]:
        raise SystemExit("congestion_labels.json does not belong to labeling/tasks.json")
    items = [i for i in labels["items"] if i["split"] == split]
    digest = hashlib.sha256(json.dumps(sorted((i["slot"], i["image"], i["label"]) for i in items)).encode()).hexdigest()
    df = pd.DataFrame(items)
    df["y"] = df["label"].map({l: k for k, l in enumerate(LEVELS)})
    return df, digest, tasks


# ---------- models ----------

def wkappa(y: np.ndarray, p: np.ndarray, k: int = 3) -> float:
    o = np.zeros((k, k))
    np.add.at(o, (y, p), 1)
    w = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    return float(1 - (w * o).sum() / (w * e).sum())


def fit_cuts(x: np.ndarray, y: np.ndarray) -> list[float]:
    qs = np.unique(np.quantile(x, np.linspace(0, 1, GRID)))
    t1, t2 = np.meshgrid(qs, qs, indexing="ij")
    pred = (x[None, None] > t1[..., None]).astype(int) + (x[None, None] > t2[..., None]).astype(int)
    acc = (pred == y).mean(-1)
    acc[t2 < t1] = -1
    i, j = np.unravel_index(np.argmax(acc), acc.shape)
    return [float(qs[i]), float(qs[j])]


def cut(x: np.ndarray, cuts: list[float]) -> np.ndarray:
    return (x > cuts[0]).astype(int) + (x > cuts[1]).astype(int)


def stack_matrix(df: pd.DataFrame, feats: dict) -> np.ndarray:
    return np.column_stack([np.log1p(df[feats["A"]]), df[feats["A_occ"]], df[feats["B"]], df[feats["C"]]])


def fit_stack(x: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.linear_model import LogisticRegression
    mu, sd = x.mean(0), x.std(0) + 1e-9
    lr = LogisticRegression(max_iter=5000).fit((x - mu) / sd, y)
    return {"mean": mu.tolist(), "scale": sd.tolist(), "coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist(),
            "inputs": ["log1p(A count)", "A occupancy", "B", "C"]}


def stack_proba(x: np.ndarray, p: dict) -> np.ndarray:
    z = ((x - np.array(p["mean"])) / np.array(p["scale"])) @ np.array(p["coef"]).T + np.array(p["intercept"])
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def predict(df: pd.DataFrame, method: str, cfg: dict) -> np.ndarray:
    feats, par = cfg["features"], cfg["methods"]
    if method in ("A", "B", "C"):
        return cut(df[feats[method]].to_numpy(float), par[method]["cuts"])
    if method == "vote":
        return np.median(np.vstack([predict(df, m, cfg) for m in ("A", "B", "C")]), 0).astype(int)
    return stack_proba(stack_matrix(df, feats), par["stack"]).argmax(1)


def cross_validate(d: pd.DataFrame, feats: dict, methods: list[str]) -> dict:
    from sklearn.model_selection import StratifiedKFold
    y = d["y"].to_numpy()
    scores = {m: {"acc": [], "kappa": []} for m in methods}
    for r in range(REPS):
        pred = {m: np.zeros(len(y), int) for m in methods}
        for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=r).split(d, y):
            cfg = fit_all(d.iloc[tr], feats, methods)
            for m in methods:
                pred[m][te] = predict(d.iloc[te], m, cfg)
        for m in methods:
            scores[m]["acc"].append(float((pred[m] == y).mean()))
            scores[m]["kappa"].append(wkappa(y, pred[m]))
    return {m: {"acc": float(np.mean(s["acc"])), "acc_sd": float(np.std(s["acc"])), "kappa": float(np.mean(s["kappa"]))}
            for m, s in scores.items()}


def fit_all(d: pd.DataFrame, feats: dict, methods: list[str]) -> dict:
    y = d["y"].to_numpy()
    par = {}
    for m in ("A", "B", "C"):
        if m in feats and (m in methods or "vote" in methods):
            par[m] = {"cuts": fit_cuts(d[feats[m]].to_numpy(float), y)}
    if "stack" in methods:
        par["stack"] = fit_stack(stack_matrix(d, feats), y)
    return {"features": feats, "methods": par}


# ---------- commands ----------

def cmd_fit(root: Path) -> None:
    lab, digest, _ = load_labels(root, "practice")
    trials = load_trials(root)
    if trials is None:
        raise SystemExit("No CLIP trials in results/congestion/clip_trials. Run first:\n  python -m src.congestion.clip_voter "
                         "--crop context mask zones --images-from results/congestion/labeling/congestion_labels.json "
                         "--out results/congestion/clip_trials")
    d = lab.merge(load_a_b(root), on=["image"], suffixes=("", "_f")).merge(trials, on="image")
    if len(d) != len(lab):
        raise SystemExit(f"Only {len(d)} of {len(lab)} practice frames have all voter scores")
    print(f"Practice labels: {len(d)} frames  {d['label'].value_counts().to_dict()}")

    # 1. feature of each voter
    groups = {"A": [c for c in d if c.startswith(("A_count", "A_pcu"))], "A_occ": ["A_occ_union", "A_occ_max"],
              "B": ["B_union", "B_max"], "C": [c for c in d if c.startswith("C_")]}
    feats, feature_cv = {}, {}
    for g, cols in groups.items():
        res = {}
        for c in cols:
            r = cross_validate(d, {"A": c}, ["A"])["A"]
            res[c] = r
            print(f"  {g:5s} {c:22s} CV accuracy {r['acc']:.3f}  kappa {r['kappa']:.3f}")
        feats[g] = max(res, key=lambda c: res[c]["kappa"])
        feature_cv[g] = res
    print("Chosen features:", feats)

    # 2. candidate methods
    cv = cross_validate(d, feats, ORDER)
    best = max(cv[m]["kappa"] for m in ORDER)
    primary = next(m for m in ORDER if cv[m]["kappa"] >= best - 0.005)
    for m in ORDER:
        print(f"  {m:6s} CV accuracy {cv[m]['acc']:.3f} (sd {cv[m]['acc_sd']:.3f})  kappa {cv[m]['kappa']:.3f}"
              f"{'   <- primary' if m == primary else ''}")

    cfg = fit_all(d, feats, ORDER)
    cfg.update({"format": "congestion_ensemble/v1", "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "primary": primary, "levels": LEVELS, "practice_labels_sha256": digest, "n_practice": len(d),
                "cv": {"folds": FOLDS, "repeats": REPS, "methods": cv, "features": feature_cv},
                "clip_crop": feats["C"].split("_")[1]})
    (root / "ensemble_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Frozen -> {root / 'ensemble_config.json'} (primary: {primary})")


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    out = {"accuracy": float((p == y).mean()), "kappa": wkappa(y, p)}
    f1 = []
    for k in range(3):
        tp, fp, fn = ((p == k) & (y == k)).sum(), ((p == k) & (y != k)).sum(), ((p != k) & (y == k)).sum()
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1.append(2 * prec * rec / max(prec + rec, 1e-9))
        if k == 2:
            out.update({"congested_precision": float(prec), "congested_recall": float(rec), "congested_f1": float(f1[-1])})
    out["macro_f1"] = float(np.mean(f1))
    return out


def cmd_test(root: Path, force: bool, boots: int = 2000) -> None:
    cfg_path, out_path = root / "ensemble_config.json", root / "eval_test.json"
    if not cfg_path.exists():
        raise SystemExit("Run `python -m src.congestion.ensemble_eval fit` first")
    if out_path.exists() and not force:
        raise SystemExit(f"{out_path} exists: the test set has already been used. Pass --force only to reproduce it.")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    _, practice_digest, _ = load_labels(root, "practice")
    if practice_digest != cfg["practice_labels_sha256"]:
        raise SystemExit("The practice labels changed after the config was fitted; fit again first")
    lab, digest, tasks = load_labels(root, "test")
    d = lab.merge(load_a_b(root), on="image", suffixes=("", "_f")).merge(load_trials(root), on="image")
    if len(d) != len(lab):
        raise SystemExit(f"Only {len(d)} of {len(lab)} test frames have all voter scores")
    y = d["y"].to_numpy()

    # population weights: frames in the camera x stratum / test frames drawn from it
    sizes = tasks["strata"]["frames_per_camera"]
    drawn = d.groupby(["camera", "stratum"]).size()
    w = np.array([sizes[c][s] / drawn[(c, s)] for c, s in zip(d["camera"], d["stratum"])])

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(y), size=(boots, len(y)))
    preds = {m: predict(d, m, cfg) for m in ORDER}
    result = {"format": "congestion_eval/v1", "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "test_labels_sha256": digest, "n_test": len(d), "primary": cfg["primary"], "features": cfg["features"],
              "bootstrap": boots, "methods": {}}
    rows = []
    for m, p in preds.items():
        r = metrics(y, p)
        for key in ("accuracy", "kappa"):
            vals = [metrics(y[i], p[i])[key] for i in idx[:boots]]
            r[f"{key}_ci95"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        r["accuracy_population_weighted"] = float((w * (p == y)).sum() / w.sum())
        r["confusion"] = pd.crosstab(pd.Series(y, name="human"), pd.Series(p, name="method")).reindex(
            index=range(3), columns=range(3), fill_value=0).to_numpy().tolist()
        result["methods"][m] = r
        for cam, g in d.assign(p=p).groupby("camera"):
            rows.append({"method": m, "group": cam, "n": len(g), "accuracy": round(float((g.p == g.y).mean()), 3)})
        for day, g in d.assign(p=p).groupby("is_day"):
            rows.append({"method": m, "group": "day" if day else "night", "n": len(g), "accuracy": round(float((g.p == g.y).mean()), 3)})
    prim, base = preds[cfg["primary"]], preds["A"]
    diff = [float((prim[i] == y[i]).mean() - (base[i] == y[i]).mean()) for i in idx]
    result["primary_minus_A_accuracy"] = {"value": float((prim == y).mean() - (base == y).mean()),
                                          "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(root / "eval_test_by_group.csv", index=False)

    print(f"Test labels: {len(d)} frames  {d['label'].value_counts().to_dict()}  (primary: {cfg['primary']})")
    for m in ORDER:
        r = result["methods"][m]
        print(f"  {m:6s} accuracy {r['accuracy']:.3f} [{r['accuracy_ci95'][0]:.3f}, {r['accuracy_ci95'][1]:.3f}]  "
              f"kappa {r['kappa']:.3f} [{r['kappa_ci95'][0]:.3f}, {r['kappa_ci95'][1]:.3f}]  "
              f"congested F1 {r['congested_f1']:.3f}  population-weighted accuracy {r['accuracy_population_weighted']:.3f}")
    dd = result["primary_minus_A_accuracy"]
    print(f"  primary - A: {dd['value']:+.3f} [{dd['ci95'][0]:+.3f}, {dd['ci95'][1]:+.3f}]")
    print(f"Wrote {out_path} and {root / 'eval_test_by_group.csv'}")


def cmd_apply(root: Path) -> None:
    cfg = json.loads((root / "ensemble_config.json").read_text(encoding="utf-8"))
    f = load_a_b(root)
    crop, needs_c = cfg["clip_crop"], cfg["primary"] in ("C", "vote", "stack")
    run_info = root / "clip_run.json"
    c_ok = (root / "voter_clip.csv").exists() and run_info.exists() and \
        json.loads(run_info.read_text(encoding="utf-8")).get("crop_mode") == crop
    if c_ok:
        f = f.merge(load_c(root / "voter_clip.csv", crop), on="image", how="left")
        if needs_c:
            ccols = [c for c in f if c.startswith("C_")]
            missing = int(f[ccols].isna().any(axis=1).sum()) if ccols else len(f)
            if missing:
                raise SystemExit(
                    f"The CLIP file covers only {len(f) - missing} of {len(f)} frames. "
                    "Run a full clip_voter pass before applying the primary method:\n"
                    f"  python -m src.congestion.clip_voter --crop {crop}"
                )
    elif needs_c:
        raise SystemExit(f"The primary method uses C with the '{crop}' crop. Score every frame first:\n"
                         f"  python -m src.congestion.clip_voter --crop {crop}")
    out = f[["image", "camera_id", "captured_at_sgt", "weekday", "hour", "is_day"]].copy()
    for m in ORDER:
        if m in ("A", "B") or c_ok:
            out[f"level_{m}"] = predict(f, m, cfg)
    out["level"] = out[f"level_{cfg['primary']}"]
    out["level_name"] = out["level"].map(dict(enumerate(LEVELS)))
    out.to_csv(root / "congestion_levels.csv", index=False)
    share = out.groupby("camera_id")["level_name"].value_counts(normalize=True).unstack(fill_value=0).round(3)
    print(share.to_string())
    print(f"Wrote {len(out)} frames -> {root / 'congestion_levels.csv'} (primary: {cfg['primary']})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Congestion levels from the three voters: fit on practice, test once, apply")
    ap.add_argument("command", choices=["fit", "test", "apply"])
    ap.add_argument("--root", type=Path, default=Path("results/congestion"))
    ap.add_argument("--force", action="store_true", help="test: overwrite an existing test result")
    args = ap.parse_args()
    {"fit": lambda: cmd_fit(args.root), "test": lambda: cmd_test(args.root, args.force),
     "apply": lambda: cmd_apply(args.root)}[args.command]()


if __name__ == "__main__":
    main()
