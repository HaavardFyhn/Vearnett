"""Build an explicit allowlisted server package, without local config or credentials."""
from pathlib import Path, PureWindowsPath
import argparse
import json
import shutil
import tempfile

import train_senorge_grid_pairs as training
import geopandas as gpd
import numpy as np

from remote_senorge.train_remote import sha256


ROOT = Path(__file__).resolve().parent


def prepare(source: Path, output: Path, target_metric: str = "rmse",
            archive_dir: Path | None = None) -> None:
    if output.exists() or Path(str(output) + ".zip").exists():
        raise FileExistsError(f"Package already exists: {output}")
    if target_metric not in ("rmse", "absolute-error-p95"):
        raise ValueError("Unknown seNorge pair target metric")
    config = json.loads((source / "config.json").read_text())
    grid = gpd.read_file(source / "grid_features.gpkg")
    pairs = np.load(source / "cache" / "pairs.npy", allow_pickle=False)
    if pairs.shape != (config["pairs"], 2):
        raise ValueError("Wrong number of cached pairs")
    plan = training.date_plan(config["years"], config["date_step"], config.get("days_per_month", 0),
                              config.get("temporal_split", "chronological"), config["seed"])
    days = len(plan)
    split = int((plan.partition == 0).sum())
    plan_path = source / "cache" / "date_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != dict(dates=plan.to_dict("records")):
        raise ValueError("Cached moment date plan does not match configuration")
    if config.get("days_per_month", 0) and not plan_path.exists():
        raise ValueError("Random-date moments require a saved date plan")
    if target_metric == "rmse":
        totals = np.zeros((2, 6, len(pairs)), dtype=float)
        for year in config["years"]:
            path = source / "cache" / f"seNorge2018_{year}.npz"
            with np.load(path, allow_pickle=False) as saved:
                moments = saved["moments"]
                if moments.shape != totals.shape:
                    raise ValueError(f"Invalid annual moments: {year}")
                totals += moments
        labels, _ = training.regression_labels(totals[0], totals[1])
        keep = (np.isfinite(labels) & (totals[0, 0] >= max(60, np.ceil(split*.9))) &
                (totals[1, 0] >= max(20, np.ceil((days-split)*.9))))
    else:
        archives = ([Path(item["path"]) for item in config["archives"]]
                    if archive_dir is None else
                    [archive_dir / PureWindowsPath(item["path"]).name for item in config["archives"]])
        missing_archives = [str(path) for path in archives if not path.is_file()]
        if missing_archives:
            raise FileNotFoundError("Missing annual seNorge archives: " + ", ".join(missing_archives))
        label_frame = training.stream_labels(
            archives, grid, pairs, source / "cache", config["date_step"], plan=plan,
            target_metric=target_metric)
        labels = label_frame.target_error.to_numpy()
        keep = np.isfinite(labels)
    pairs, labels = pairs[keep], labels[keep]
    x = training.pair_features(grid, pairs).to_numpy()
    train, test = training.spatial_split(grid, pairs)
    distance = x[:, training.FEATURES.index("distance_m")]
    spatial_distance_bins = []
    for lower, upper in ((1, 5), (5, 10), (10, 20), (20, 30), (30, 50),
                         (50, 75), (75, 90), (90, 100)):
        mask = (distance >= lower*1000) & (
            distance <= upper*1000 if upper == 100 else distance < upper*1000)
        spatial_distance_bins.append(dict(lower_km=lower, upper_km=upper,
                                          all_pairs=int(mask.sum()),
                                          spatial_train=int((mask & train).sum()),
                                          spatial_test=int((mask & test).sum())))
    _, blocks = np.unique(np.floor(grid[["x", "y"]].to_numpy()/100_000).astype(int),
                          axis=0, return_inverse=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="server_package_") as temporary:
        work = Path(temporary)
        np.savez_compressed(work / "training.npz", x=x, y=labels.astype("float32"), train=train, test=test,
                            grid_id_1=grid.grid_id.to_numpy()[pairs[:, 0]],
                            grid_id_2=grid.grid_id.to_numpy()[pairs[:, 1]],
                            block_1=blocks[pairs[:, 0]], block_2=blocks[pairs[:, 1]])
        for name in ("train_remote.py", "requirements.txt", "SERVER_README.md", "test_train_remote.py"):
            shutil.copyfile(ROOT / "remote_senorge" / name, work / name)
        plan.to_csv(work / "selected_dates.csv", index=False)
        manifest = dict(
            schema=1, dataset_sha256=sha256(work / "training.npz"), features=training.FEATURES,
            target_metric=target_metric,
            base_feature_count=len(training.PAIR_FEATURES), pairs=len(labels),
            spatial_train=int(train.sum()), spatial_test=int(test.sum()),
            years=config["years"], sampled_days=days, train_days=split, test_days=days-split,
            temporal_split=config.get("temporal_split", "chronological"),
            days_per_month=config.get("days_per_month", 0),
            maximum_distance_m=config["maximum_distance_m"],
            distance_distribution=config.get("distance_distribution", "gaussian-offset"),
            distance_scale_m=config.get("distance_scale_m", 15_000),
            selected_dates_sha256=sha256(work / "selected_dates.csv"),
            sampling=(json.loads((source / "sampling.json").read_text())
                      if (source / "sampling.json").exists() else None),
            spatial_distance_bins=spatial_distance_bins,
            training_source="seNorge_2018 only; no Frost station pairs or station-based sampling",
            target=(config["target"] if target_metric == "rmse" else
                    "Mean of the two directional 95th percentiles of absolute OLS residuals; " +
                    config.get("temporal_split", "chronological")),
            lake_semantics=config["lake_semantics"],
            limitations=["Gridded interpolation similarity, not independent station accuracy.",
                         "Basins are terrain depressions, not observed lakes.",
                         "Whole-basin volumes repeat in intersecting cells; not additive national volume.",
                         f"Trained distance domain 1-{config['maximum_distance_m']/1000:g} km; not validated outside this interval.",
                         "Rare long distances may have few or no spatial holdout pairs; inspect spatial_distance_bins before claiming accuracy.",
                         "Random dates within years measure historical similarity, not future-period performance."
                         if config.get("temporal_split") == "monthly-random"
                         else "Temporal label validation uses the final 20% of sampled dates.",
                         "No direct Station Lab integration; keep existing profiles unchanged."],
        )
        training.write_json(work / "manifest.json", manifest)
        # Per-cell covariates needed later for inference, not raw geospatial files/config.
        grid.drop(columns="geometry").to_csv(work / "grid_features.csv.gz", index=False, compression="gzip")
        allowed = sorted(p.name for p in work.iterdir())
        training.write_json(work / "package_checksums.json", {p.name: sha256(p) for p in work.iterdir()})
        shutil.make_archive(str(output), "zip", work)
    print(f"Prepared {output}.zip; allowlisted members: {allowed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive-dir", type=Path,
                        help="Directory holding the annual seNorge NetCDFs used for p95 labels")
    parser.add_argument("--prepare-30yr", action="store_true",
                        help="Prepare 1m half-normal pairs and 3600 random monthly dates, then package")
    parser.add_argument("--target-metric", choices=["rmse", "absolute-error-p95"], default="rmse",
                        help="Pair label; absolute-error-p95 changes only the 30-year seNorge target")
    args = parser.parse_args()
    if args.source is None:
        args.source = training.WORK / ("senorge_grid_pairs_1m_1996_2025_random" if args.prepare_30yr
                                       else "senorge_grid_pairs_1m_basins")
    if args.output is None:
        suffix = "_p95" if args.target_metric == "absolute-error-p95" else ""
        args.output = training.WORK / (
            ("senorge_server_package_1996_2025" + suffix) if args.prepare_30yr
            else "senorge_server_package" + suffix)
    if args.output.with_suffix(".zip").exists():
        raise FileExistsError("Package ZIP already exists; choose a new --output")
    if args.prepare_30yr:
        preparation_args = [
            "--output-dir", str(args.source), "--pairs", "1000000", "--prepare-only",
            "--years", *map(str, range(1996, 2026)), "--days-per-month", "10",
            "--temporal-split", "monthly-random", "--distance-distribution", "half-normal",
            "--distance-scale-m", "25000", "--maximum-distance-m", "100000",
        ]
        if args.archive_dir is None:
            preparation_args.extend([
                "--archive-dir", str(training.WORK / "senorge_norway_1996_2025" / "netcdf"),
                "--reuse-archive-dir", str(training.WORK / "senorge_norway_2021_2025" / "netcdf"),
            ])
        else:
            preparation_args.extend(["--archive-dir", str(args.archive_dir)])
        training.main(preparation_args)
    config = json.loads((args.source / "config.json").read_text())
    if args.target_metric == "absolute-error-p95" and (
            config["years"] != list(range(1996, 2026)) or
            config.get("days_per_month") != 10 or
            config.get("temporal_split") != "monthly-random" or
            config.get("distance_distribution") != "half-normal" or
            config.get("distance_scale_m") != 25_000 or
            config.get("maximum_distance_m") != 100_000):
        parser.error("The p95 target requires the existing 1996-2025 seNorge30 sample")
    prepare(args.source, args.output, args.target_metric, archive_dir=args.archive_dir)


if __name__ == "__main__":
    main()
