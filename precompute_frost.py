"""Forhåndsberegn eksakte Frost-grunnflater separat, før kartappen brukes."""
from contextlib import contextmanager
import argparse
import os
from pathlib import Path

import pandas as pd
import pyogrio

from greedy_400_stations_pruned import make_cache
from make_uncertainty_map_gif import HERE, OUT, SOURCE_GRID, load_grid_all_fields
from station_weather import PHENOMENON_DATASETS, WEATHER, WeatherRegistry, restricted_profile
from station_weather_maps import DISPLAY_SIZES, WeatherMapCache, colour_reference, frost_limits


@contextmanager
def computation_lock(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "precompute_frost.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        acquired = False
        try:
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError:
                raise RuntimeError("En annen separat Frost-forhåndsberegning kjører allerede") from None
            yield
        finally:
            if acquired:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phenomenon", nargs="+", choices=WEATHER, default=list(WEATHER),
                        help="Fenomenene som skal beregnes; standard er alle")
    parser.add_argument("--cell-m", "--grid-spacing-m", nargs="+", type=int,
                        choices=[5000, 2500, 1000, 500], default=[5000, 2500, 1000])
    parser.add_argument("--kind", choices=["continuous", "class", "senorge", "senorge30", "senorge30_pross",
                        "seasonal_winter_tg", "seasonal_summer_tg", "seasonal_winter_tn", "seasonal_summer_tn",
                        "seasonal", "all"], default="all",
                        help="Choose senorge explicitly for the experimental local-pair profile")
    parser.add_argument("--all-weather-maps", action="store_true",
                        help="Prepare surfaces and PNGs for the Station Lab models (add --include-experimental for all)")
    parser.add_argument("--prepare-maps", action="store_true",
                        help="Also save fixed weather PNGs at 400, 800 and 1600 pixels")
    parser.add_argument("--include-experimental", action="store_true",
                        help="Also include models not offered in Station Lab (seNorge pilot, CNN, GPU, other kinds)")
    parser.add_argument("--cached-only", action="store_true",
                        help="Only prepare images for existing surfaces; never predict missing surfaces")
    parser.add_argument("--station-network", choices=["all", "svv", "both"], default="both",
                        help="Prepare all stations, SVV only, or both (default: both)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Vis tilgjengelige jobber uten beregning eller skriving")
    args = parser.parse_args(argv)
    if args.kind == "senorge30_pross":
        args.kind = "senorge30"
    args.cell_m = sorted(set(args.cell_m), reverse=True)
    if args.all_weather_maps:
        args.prepare_maps = True
    return args


def selected_modes(registry, args):
    used = set(PHENOMENON_DATASETS.values())
    modes = [mode for mode in registry.available if mode["phenomenon"] in args.phenomenon and
             ((args.kind == "all" and (mode["id"] in used or getattr(args, "include_experimental", False)))
              or mode["kind"] == args.kind
              or (args.kind == "seasonal" and mode["kind"].startswith("seasonal_")))]
    missing = set(args.phenomenon) - {mode["phenomenon"] for mode in modes}
    if missing and (args.kind != "all" or not modes):
        raise ValueError("Ingen ferdig modell for valgt prediksjonsmål: " + ", ".join(sorted(missing)))
    if missing:
        print("Hopper over fenomen uten ferdig modell: " + ", ".join(sorted(missing)), flush=True)
    return modes


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        for spacing in args.cell_m:
            registry = WeatherRegistry(HERE, None, None, None, cell_m=spacing)
            for mode in selected_modes(registry, args):
                print(f"{spacing} m · {mode['label']} · {mode['metric']} · {mode['prefix']} · {args.station_network}")
        print("Tørrkjøring: ingen filer skrevet og ingen grunnflater beregnet")
        return 0

    for spacing in args.cell_m:
        if not (OUT / f"norway_grid_{spacing}m.gpkg").exists():
            raise FileNotFoundError(f"Mangler terrengrutenett for {spacing} m")
        selected_modes(WeatherRegistry(HERE, None, None, None, cell_m=spacing), args)

    models = {}
    missing = []
    images = WeatherMapCache(OUT / "station_app_cache" / "weather_maps") if args.prepare_maps else None
    with computation_lock(OUT / "station_app_cache"):
        source_types = pyogrio.read_dataframe(OUT / f"norway_grid_{min(1000, *args.cell_m)}m.gpkg",
                                              columns=["objtype"], read_geometry=False)["objtype"].dropna().unique()
        original = load_grid_all_fields(SOURCE_GRID)
        for spacing in args.cell_m:
            grid = original.copy() if spacing == 5000 else load_grid_all_fields(OUT / f"norway_grid_{spacing}m.gpkg")
            categories = pd.Index(original["objtype"].unique()).union(grid["objtype"].unique()).union(source_types)
            grid["objtype_code"] = pd.Categorical(grid["objtype"], categories=categories).codes
            registry = WeatherRegistry(HERE, grid, make_cache(grid), categories, models=models,
                                       cell_m=spacing, allow_precompute=not args.cached_only)
            renderer = None
            if args.prepare_maps:
                from station_app import StationEngine
                renderer = StationEngine.__new__(StationEngine)
                renderer.cell_m = spacing
                renderer.grid = grid
                renderer.norway_path = OUT / "norway_polygon.gpkg"
                renderer.minimum, renderer.maximum = 0.5, 6.5
                renderer._prepare_raster()
            for mode in selected_modes(registry, args):
                print(f"START {spacing} m · {mode['label']} · {mode['metric']}", flush=True)
                profile = registry.profile(mode["id"])
                if profile["status"]["state"] != "ready":
                    if args.cached_only:
                        missing.append((spacing, mode["id"]))
                        print(f"MANGLER {spacing} m · {mode['id']}; hoppes over uten beregning", flush=True)
                        continue
                    raise RuntimeError("Forhåndsberegningen ble ikke fullført")
                networks = [("all", profile)] if args.station_network in ("all", "both") else []
                reference = None
                if args.station_network in ("svv", "both") or args.prepare_maps:
                    from station_app import pross_excluded
                    excluded = pross_excluded(profile["stations"])
                    svv = restricted_profile(profile, registry.target_cache, excluded, build=not args.cached_only)
                    if svv is None:
                        if len(excluded) == len(profile["stations"]):
                            print(f"SVV TOMT {spacing} m · {mode['id']}; ingen kvalifiserte stasjoner", flush=True)
                            if profile["phenomenon"] == "precipitation":
                                reference = registry.svv_profile(mode["id"])
                                print(reference["colour_reference_warning"], flush=True)
                        else:
                            missing.append((spacing, mode["id"] + ":svv"))
                            print(f"MANGLER SVV {spacing} m · {mode['id']}; hoppes over uten beregning", flush=True)
                    else:
                        reference = svv
                        if args.station_network in ("svv", "both"):
                            networks.append(("svv", svv))
                for network, prepared in networks:
                    if renderer is not None and reference is not None:
                        colour_profile = (colour_reference(registry, mode["id"], profile)
                                          if profile["phenomenon"] == "road_temperature" else reference)
                        for size in DISPLAY_SIZES:
                            images.get(renderer, prepared, size, frost_limits(colour_profile), build=True)
                        print(f"PNG KLAR {spacing} m · {mode['id']} · {network} · 400/800/1600 px", flush=True)
                    print(f"NETT KLAR {network} · {prepared['cache_path']}", flush=True)
                print(f"KLAR {profile['cache_path']}", flush=True)
    if missing:
        print(f"Mangler fortsatt {len(missing)} grunnflater; kjør uten --cached-only for å beregne dem.", flush=True)
    print("Alle valgte eksakte Frost-grunnflater er lagret. Kartappen kan nå startes.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())