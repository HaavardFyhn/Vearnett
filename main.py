import argparse
from pipeline_config import PHENOMENA, BASE_GRID_FILE, GRID_FILE
from pathlib import Path
STEPS = [ "norway_polygon", "stations", "download", "trainmap", "information", ]


def parse_args():
    p=argparse.ArgumentParser(description="Weather-station prediction-error pipeline v3")
    p.add_argument("--phenomenon",choices=PHENOMENA,default="temperature")
    p.add_argument("--years",type=int,default=1)
    p.add_argument("--from-step",choices=STEPS,default="download")
    p.add_argument("--to-step",choices=STEPS,default="information")
    p.add_argument("--force",action="store_true")
    p.add_argument("--force-download",action="store_true")
    p.add_argument("--skip-download",action="store_true")
    p.add_argument("--grid-spacing-m", "--grid-spacing", type=int, default=5000, help="Trainmap spacing in meters")
    p.add_argument("--output-dir", type=Path, default=None, help="Separate analysis output directory")
    p.add_argument("--station-cell-inputs", action="store_true",
                   help="Train using canonical 500/1000 m cell coordinates, elevation and terrain for stations")
    p.add_argument("--metric", choices=[ "prediction_rmse", "temperature_class_error", "precipitation_occurrence_error", "snow_occurrence_error" ], default="prediction_rmse" )
    return p.parse_args()

def selected(name,a): return STEPS.index(a.from_step)<=STEPS.index(name)<=STEPS.index(a.to_step)
def grid_file(spacing_m): return ( GRID_FILE.parent / f"norway_grid_{spacing_m}m.gpkg" )

def main():
    a=parse_args()
    if a.years<1: raise ValueError("--years must be at least 1")
    if a.phenomenon == "humidity" and a.metric != "prediction_rmse":
        raise ValueError("Humidity currently supports --metric prediction_rmse only; humidity classes are not defined")
    if STEPS.index(a.from_step)>STEPS.index(a.to_step): raise ValueError("--from-step must precede --to-step")
    if selected("norway_polygon",a):
        from norway_polygon import build as build_norway_polygon; build_norway_polygon(force=a.force)
    else: print("SKIP norway_polygon")
    if selected("stations",a):
        from fetch_stations import run as fetch_stations_run; fetch_stations_run()
    else: print("SKIP stations")
    if selected("download",a) and not a.skip_download:
        from download_frost import run as download_run; download_run(a.phenomenon,a.years,force=(a.force or a.force_download))
    else: print("SKIP download")
    if selected("trainmap", a):
        current_grid = grid_file(a.grid_spacing_m)
        if not current_grid.exists() or a.force:
            from build_trainmap import build
            build(spacing_m=a.grid_spacing_m, force=a.force)
        else:
            print(f"SKIP trainmap ({current_grid.name})")
    else: print("SKIP trainmap")
    if selected("information",a):
        current_grid = grid_file( a.grid_spacing_m )
        if not current_grid.exists(): raise FileNotFoundError(f"Missing terrain grid: {current_grid}")
        from information_gain import run
        run(a.phenomenon, a.years, metric=a.metric, force=a.force,
            grid_file=current_grid, output_dir=a.output_dir, cell_inputs=a.station_cell_inputs)
    print("Pipeline complete")
if __name__=="__main__": main()
