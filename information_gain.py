import json, os, sqlite3, time
from contextlib import closing
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from shapely.prepared import prep
from pipeline_config import *
from scipy.spatial import cKDTree
from calculation_progress import report as report_progress

EXPERIMENT_TAG = "radius150km"
TERRAIN = ["relief_10km","rugstd_10km","tpi_10km","relief_25km","rugstd_25km","tpi_25km"]



def paths(name, years, metric, radius=None, output_dir=None):

    tag = f"{name}_{years}y_{metric}"
    output = WORK if output_dir is None else Path(output_dir)

    suffix = (
        f"_radius{radius//1000}km"
        if radius is not None
        else ""
    )

    return {
        "stations": output / f"{tag}_station_features.gpkg",
        "pairs": output / f"{tag}_pairs.csv",
        "metadata": output / f"{tag}_metadata.json",
        "model": output / f"{tag}_model.joblib",

        "surface":
            output /
            f"{tag}_error_reduction{suffix}.gpkg",

        "top":
            output /
            f"{tag}_top100_error_reduction{suffix}.gpkg"
    }

def clean(g):
    return g.drop(columns=[c for c in ["index_left","index_right"] if c in g.columns],errors="ignore")

def temp_class(t):
    if t < -4:
        return 0
    if t <= 4:
        return 1
    return 2


def load_grid(grid_file):
    g=clean(pyogrio.read_dataframe(grid_file, layer="trainmap"))
    if g.crs is None: raise ValueError("Grid CRS is missing")
    if g.crs.to_epsg()!=25833: g=g.to_crs(25833)
    required = [ "elevation", "coast_dist_fjord", "coast_dist_ocean", "objtype", *TERRAIN, "geometry"]
    missing=set(required)-set(g.columns)
    if missing: raise ValueError(f"Grid missing fields: {sorted(missing)}")
    g=g.dropna(subset=required[:-1]).copy()
    g["x"],g["y"]=g.geometry.x,g.geometry.y
    g["grid_id"]=np.arange(len(g),dtype=np.int32)
    g["coast_dist"] = g["coast_dist_fjord"]
    return g

def station_features(grid,p,force=False,cell_inputs=False):
    if p["stations"].exists() and not force and not cell_inputs:
        s=pyogrio.read_dataframe(p["stations"])
        if s.crs.to_epsg()!=25833: s=s.to_crs(25833)
        return s
    with closing(sqlite3.connect(DATABASE.resolve().as_uri() + "?mode=ro", uri=True)) as cx:
        s=pd.read_sql("SELECT station_id,lat,lon,elevation FROM stations",cx)
    s=gpd.GeoDataFrame(s,geometry=gpd.points_from_xy(s.lon,s.lat),crs=4326).to_crs(25833)
    if cell_inputs:
        valid = np.isfinite(np.column_stack([s.geometry.x, s.geometry.y])).all(axis=1)
        if not valid.all():
            print(f"Excluding {int((~valid).sum())} stations with missing/invalid coordinates before 1 km assignment",
                  flush=True)
            s = s.loc[valid].copy()
    s = gpd.sjoin_nearest( clean(s), grid[ [ "coast_dist_fjord", "coast_dist_ocean", "objtype", *TERRAIN, "geometry" ] ], how="left" )
    s["objtype"] = (
        s["objtype"]
        .astype("category")
    )

    if "elevation_left" in s: s["elevation"]=s["elevation_left"]
    s["x"],s["y"]=s.geometry.x,s.geometry.y
    if cell_inputs:
        from station_senorge import station_cell_features
        s = station_cell_features(s, grid)
        s.geometry = gpd.points_from_xy(s.x, s.y, crs=25833)
    s=clean(s).dropna(subset=["station_id","x","y","elevation","coast_dist_fjord","coast_dist_ocean","objtype",*TERRAIN]).drop_duplicates("station_id")
    p["stations"].parent.mkdir(parents=True,exist_ok=True)
    pyogrio.write_dataframe(s,p["stations"],driver="GPKG")
    return s

def directional_rmse(train,valid,source_col,target_col):
    reg=LinearRegression().fit(train.iloc[:,[source_col]],train.iloc[:,target_col])
    pred=reg.predict(valid.iloc[:,[source_col]])
    return np.sqrt(mean_squared_error(valid.iloc[:,target_col],pred))

def directional_class_error(
    train,
    valid,
    source_col,
    target_col
):

    reg = LinearRegression()

    reg.fit(
        train.iloc[:, [source_col]],
        train.iloc[:, target_col]
    )

    pred = reg.predict(
        valid.iloc[:, [source_col]]
    )

    actual_class = np.array(
        [
            temp_class(v)
            for v in valid.iloc[:, target_col]
        ]
    )

    pred_class = np.array(
        [
            temp_class(v)
            for v in pred
        ]
    )

    return (
        1 -
        accuracy_score(
            actual_class,
            pred_class
        )
    )


def directional_precip_occurrence_error(train, valid, source_col, target_col):
    # Predict wet probability from the continuous source precipitation amount.
    y_train=(train.iloc[:,target_col].to_numpy(float)>=PRECIP_WET_THRESHOLD_MM).astype(np.int8)
    y_valid=(valid.iloc[:,target_col].to_numpy(float)>=PRECIP_WET_THRESHOLD_MM).astype(np.int8)
    x_train=np.log1p(train.iloc[:,[source_col]].clip(lower=0).to_numpy(float))
    x_valid=np.log1p(valid.iloc[:,[source_col]].clip(lower=0).to_numpy(float))
    if np.unique(y_train).size < 2:
        probability_wet=np.full(len(valid),y_train.mean(),dtype=float)
    else:
        classifier=LogisticRegression(max_iter=1000,class_weight="balanced",random_state=RANDOM_STATE)
        classifier.fit(x_train,y_train)
        probability_wet=classifier.predict_proba(x_valid)[:,1]
    return brier_score_loss(y_valid,np.clip(probability_wet,0.0,1.0))

def directional_snow_occurrence_error(
    train,
    valid,
    source_col,
    target_col
):
    y_train = (
        train.iloc[:, target_col]
        .to_numpy(float)
        >= SNOW_PRESENT_THRESHOLD_CM
    ).astype(np.int8)

    y_valid = (
        valid.iloc[:, target_col]
        .to_numpy(float)
        >= SNOW_PRESENT_THRESHOLD_CM
    ).astype(np.int8)

    x_train = np.log1p(
        train.iloc[:, [source_col]]
        .clip(lower=0)
        .to_numpy(float)
    )

    x_valid = np.log1p(
        valid.iloc[:, [source_col]]
        .clip(lower=0)
        .to_numpy(float)
    )

    if np.unique(y_train).size < 2:

        probability_snow = np.full(
            len(valid),
            y_train.mean(),
            dtype=float
        )

    else:

        classifier = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        )

        classifier.fit(
            x_train,
            y_train
        )

        probability_snow = (
            classifier.predict_proba(
                x_valid
            )[:, 1]
        )

    probability_snow = np.clip(
        probability_snow,
        0.0,
        1.0
    )

    return brier_score_loss(
        y_valid,
        probability_snow
    )


def build_pairs( name, years, stations, p, metric, force=False,cell_inputs=False,cell_m=1000 ):
    cfg=PHENOMENA[name]
    expected = {
        "target_type": metric,
        "features": PAIR_FEATURES,

        "train_fraction": TRAIN_FRACTION,
        "min_valid_days": MIN_VALID_DAYS,

        "min_overlap_days":
            cfg["min_overlap_days"] * years,

        "years": years,

        "phenomenon": name,
        "observation_update_ns": (
            element_update_marker(DATABASE, cfg["element"]).stat().st_mtime_ns
            if element_update_marker(DATABASE, cfg["element"]).exists() else None
        ),

        "precip_wet_threshold_mm": (
            PRECIP_WET_THRESHOLD_MM
            if metric == "precipitation_occurrence_error"
            else None
        ),
        "snow_present_threshold_cm": ( SNOW_PRESENT_THRESHOLD_CM if metric == "snow_occurrence_error" else None ),
        "temperature_class_limits": (
            [-4.0, 4.0]
            if metric == "temperature_class_error"
            else None
        )
    }
    if cell_inputs:
        from grid_resolution import cell_policy
        expected["station_input_policy"] = cell_policy(cell_m)

    if p["pairs"].exists() and p["metadata"].exists() and not force:
        if json.loads(p["metadata"].read_text())==expected:
            pairs=pd.read_csv(p["pairs"])
            ids=pd.Index(pd.unique(pairs[["station_1","station_2"]].to_numpy().ravel()))
            st=stations.drop_duplicates("station_id").set_index("station_id")
            ids=ids.intersection(st.index)
            return pairs,st.loc[ids]
    with closing(sqlite3.connect(DATABASE.resolve().as_uri() + "?mode=ro", uri=True)) as cx:
        obs=pd.read_sql("SELECT station_id,reference_time,value FROM observations WHERE element=?",cx,params=[cfg["element"]])
    if obs.empty:
        raise ValueError(f"No observations for {name} ({cfg['element']}); download data before running information analysis")
    obs["reference_time"]=pd.to_datetime(obs.reference_time,format="mixed",utc=True,errors="coerce")
    obs["value"]=pd.to_numeric(obs.value,errors="coerce")
    obs=obs.dropna(subset=["reference_time","value"]); obs["date"]=obs.reference_time.dt.floor("D")
    daily=getattr(obs.groupby(["date","station_id"])["value"],cfg["daily_agg"])().reset_index()
    pivot=daily.pivot(index="date",columns="station_id",values="value").sort_index()
    st=stations.drop_duplicates("station_id").set_index("station_id")
    min_obs=cfg["min_days_per_year"]*years
    ids=pivot.count()[lambda x:x>=min_obs].index.intersection(st.index)
    if len(ids) < 2:
        raise ValueError(f"Need at least two {name} stations with {min_obs} valid daily observations; found {len(ids)}")
    pivot=pivot.loc[:,ids]; st=st.loc[ids]
    available=pivot.notna().astype(np.uint16); overlap=available.T.dot(available)
    i,j=np.triu_indices(len(ids),1); ov=overlap.to_numpy()[i,j]
    min_overlap=cfg["min_overlap_days"]*years; keep=ov>=min_overlap
    i,j,ov=i[keep],j[keep],ov[keep]
    target_values = []

    pair_started = time.monotonic()
    last_progress = pair_started
    for pair_number,(a,b) in enumerate(zip(i,j),1):
        now = time.monotonic()
        if pair_number == 1 or pair_number == len(i) or now-last_progress >= 10:
            progress = report_progress("Stasjonspar", pair_number-1, len(i), pair_started)
            eta = progress["remaining_seconds"]
            print(f"Pairs {pair_number-1}/{len(i)}; ETA "
                  f"{eta/60:.1f} min" if eta is not None else f"Pairs 0/{len(i)}; ETA pending", flush=True)
            last_progress = now
        values=pd.concat([pivot.iloc[:,a],pivot.iloc[:,b]],axis=1).dropna()
        split=int(len(values)*TRAIN_FRACTION); train,valid=values.iloc[:split],values.iloc[split:]
        if len(train)<min_overlap or len(valid)<MIN_VALID_DAYS: target_values.append(np.nan); continue
        if metric == "prediction_rmse":

            value_1_to_2 = directional_rmse(train,valid,0,1)

            value_2_to_1 = directional_rmse(train,valid,1,0)

        elif metric == "temperature_class_error":

            if name != "temperature":
                raise ValueError(
                    "temperature_class_error requires "
                    "--phenomenon temperature"
                )

            value_1_to_2 = directional_class_error(train,valid,0,1)

            value_2_to_1 = directional_class_error(train,valid,1,0)

        elif metric == "precipitation_occurrence_error":

            if name != "precipitation":
                raise ValueError(
                    "precipitation_occurrence_error requires "
                    "--phenomenon precipitation"
                )

            value_1_to_2 = directional_precip_occurrence_error(train,valid,0,1)

            value_2_to_1 = directional_precip_occurrence_error(train,valid,1,0)

        elif metric == "snow_occurrence_error":
            if name != "snow":
                raise ValueError(
                    "snow_occurrence_error requires "
                    "--phenomenon snow"
                )

            value_1_to_2 = (directional_snow_occurrence_error( train,valid,0,1))

            value_2_to_1 = (directional_snow_occurrence_error(train,valid,1,0))

        else:

            raise ValueError(
                f"Unknown metric: {metric}"
            )

        target_values.append((value_1_to_2+value_2_to_1) / 2)

    report_progress("Stasjonspar", len(i), len(i), pair_started)

    def a(c): return st[c].to_numpy(float)
    x = a("x")
    y = a("y")

    e = a("elevation")

    fjord = a("coast_dist_fjord")
    ocean = a("coast_dist_ocean")
    obj = ( st["objtype"] .astype(str) .to_numpy() )
    pair={"station_1":st.index.to_numpy()[i],"station_2":st.index.to_numpy()[j],
          "distance_m":np.hypot(x[i]-x[j],y[i]-y[j]),"elev_diff_m":abs(e[i]-e[j]),
          "coast_fjord_diff_m": abs(fjord[i] - fjord[j]), "coast_ocean_diff_m": abs(ocean[i] - ocean[j]),
          "x_diff":abs(x[i]-x[j]),"y_diff":abs(y[i]-y[j]),
          "same_objtype": (obj[i] == obj[j]).astype(np.int8),
          "target_error":np.array(target_values),"n_overlap":ov}
    for base in TERRAIN:
        v=a(base); pair[base.replace("_10km","_diff_10km").replace("_25km","_diff_25km")]=abs(v[i]-v[j])
    print("PAIR_FEATURES:")
    print(PAIR_FEATURES)

    print("\nPair columns:")
    print(sorted(pair.keys()))

    missing = set(PAIR_FEATURES) - set(pair.keys())

    print("\nMissing features:")
    print(missing)

    print(
    "Stations after intersection:",
    len(ids)
    )

    print(
        "Overlap pairs before filtering:",
        len(i)
    )

    print(
        "Target values:",
        len(target_values)
    )


    pairs=pd.DataFrame(pair).replace([np.inf,-np.inf],np.nan).dropna(subset=PAIR_FEATURES+["target_error"]).copy()
    if pairs.empty:
        raise ValueError(f"No valid {name} station pairs satisfy the overlap/train/validation requirements")
    pairs.to_csv(p["pairs"],index=False); p["metadata"].write_text(json.dumps(expected,indent=2))
    print(pairs.target_error.describe()); print( f"Saved {len(pairs):,} " f"{metric} pairs" )
    print(
    "Pairs after dropna:",
    len(pairs)
)

    return pairs,st

def fit_error_model(pairs,p,force=False):
    if p["model"].exists() and not force: return joblib.load(p["model"])
    train=pairs.sample(min(MAX_TRAIN_PAIRS,len(pairs)),random_state=RANDOM_STATE) if len(pairs)>MAX_TRAIN_PAIRS else pairs
    split=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=RANDOM_STATE)
    tr,te=next(split.split(train,groups=train.station_1))
    model=ExtraTreesRegressor(n_estimators=N_TREES,min_samples_leaf=MIN_SAMPLES_LEAF,max_features=1.0,n_jobs=int(os.environ.get("WEATHER_CPU_THREADS", "-1")),random_state=RANDOM_STATE)
    model.fit(
        train[PAIR_FEATURES].to_numpy(
            dtype=np.float32
        ),
        train.target_error.to_numpy(
            dtype=np.float32
        )
    )
    actual=train.iloc[te].target_error.to_numpy(); pred=np.maximum(model.predict(train.iloc[te][PAIR_FEATURES].to_numpy(np.float32)),0)
    baseline=np.full(len(actual),train.iloc[tr].target_error.mean())
    print(f"Baseline MAE={mean_absolute_error(actual,baseline):.4f}; R2={r2_score(actual,baseline):.4f}")
    print(f"Validation MAE={mean_absolute_error(actual,pred):.4f}; R2={r2_score(actual,pred):.4f}")
    for n,v in zip(PAIR_FEATURES,model.feature_importances_): print(f"  {n}: {v:.3f}")
    model.fit(
        train[PAIR_FEATURES]
            .to_numpy(np.float32),
        train.target_error
            .to_numpy(np.float32)
    )
    temporary_model = p["model"].with_suffix(".joblib.tmp")
    joblib.dump(model, temporary_model)
    temporary_model.replace(p["model"])
    return model



def current_error(
    model,
    target_cache,
    station_cache,
    metric
):
    target_ids = np.arange(len(target_cache["x"]), dtype=np.int64)
    pred = predict_error(model, station_cache, target_cache, target_ids, metric)
    return pred.min(axis=0)


def save_results(candidates,gain,improved,p,target_count):
    g=candidates.copy(); g["total_error_reduction"]=gain; g["mean_error_reduction"]=gain/target_count
    g["improved_cells_001"]=improved
    g["baseline_method"] = "minimum_single_station"
    pyogrio.write_dataframe(g,p["surface"],layer="error_reduction",driver="GPKG")
    ordered=g.sort_values("total_error_reduction",ascending=False); chosen=[]; xy=[]
    for idx,row in ordered.iterrows():
        point=np.array([row.geometry.x,row.geometry.y])
        if not xy or min(np.linalg.norm(point-z) for z in xy)>=MIN_CANDIDATE_SEPARATION_M:
            chosen.append(idx); xy.append(point)
            if len(chosen)==TOP_N: break
    top=g.loc[chosen].copy(); top["rank"]=np.arange(1,len(top)+1)
    pyogrio.write_dataframe(top,p["top"],layer="top100",driver="GPKG")


def predict_error(
    model,
    src,
    target_cache,
    target_ids,
    metric
):

    sx = src["x"][:, None]
    sy = src["y"][:, None]

    se = src["elevation"][:, None]

    sc_fjord = src["coast_dist_fjord"][:, None]
    sc_ocean = src["coast_dist_ocean"][:, None]

    tx = target_cache["x"][target_ids][None, :]
    ty = target_cache["y"][target_ids][None, :]

    te = target_cache["elevation"][target_ids][None, :]
    tc_fjord = target_cache["coast_dist_fjord"][target_ids][None, :]
    tc_ocean = target_cache["coast_dist_ocean"][target_ids][None, :]
    same_objtype = ( src["objtype_code"][:, None] == target_cache["objtype_code"][target_ids][None, :] ).astype(np.float32)



    X = np.column_stack([
        np.hypot(sx - tx, sy - ty).ravel(),
        np.abs(se - te).ravel(),
        np.abs(sc_fjord - tc_fjord).ravel(),
        np.abs(sc_ocean - tc_ocean).ravel(),
        same_objtype.ravel(),
        np.abs(sx - tx).ravel(),
        np.abs(sy - ty).ravel(),

        np.abs(
            src["relief_10km"][:, None]
            - target_cache["relief_10km"][target_ids][None, :]
        ).ravel(),

        np.abs(
            src["rugstd_10km"][:, None]
            - target_cache["rugstd_10km"][target_ids][None, :]
        ).ravel(),

        np.abs(
            src["tpi_10km"][:, None]
            - target_cache["tpi_10km"][target_ids][None, :]
        ).ravel(),

        np.abs(
            src["relief_25km"][:, None]
            - target_cache["relief_25km"][target_ids][None, :]
        ).ravel(),

        np.abs(
            src["rugstd_25km"][:, None]
            - target_cache["rugstd_25km"][target_ids][None, :]
        ).ravel(),

        np.abs(
            src["tpi_25km"][:, None]
            - target_cache["tpi_25km"][target_ids][None, :]
        ).ravel(),
    ]).astype( np.float32, copy=False )


    pred = model.predict(X)

    pred = pred.reshape(
        len(src["x"]),
        len(target_ids)
    )


    pred = np.maximum(pred,0).astype(np.float32)
    if metric in {
        "temperature_class_error",
        "precipitation_occurrence_error",
        "snow_occurrence_error"
    }:

        pred = np.clip(
            pred,
            0.0,
            1.0
        )


    return pred

def evaluate( candidate_cache, targets, model, metric, tree = None, radius = None, target_cache = None):
    gain=np.zeros(len(candidate_cache["x"])); improved=np.zeros(len(candidate_cache["x"]),dtype=np.int32); start=time.time()
    t_predict = 0



    for c0 in range(0,len(candidate_cache["x"]),CANDIDATE_BATCH):
        start_idx = c0
        end_idx = min(
            c0 + CANDIDATE_BATCH,
            len(candidate_cache["x"])
        )
        batch_cache = {
            key: value[start_idx:end_idx]
            for key, value in candidate_cache.items()
        }

        candidate_xy = np.column_stack([
            batch_cache["x"],
            batch_cache["y"]
        ])


        candidate_nearby = []

        total_pairs = 0
        for cxy in candidate_xy:

            if tree is not None and radius is not None:

                nearby_ids = tree.query_ball_point(
                    cxy,
                    r=radius
                )

            else:

                nearby_ids = np.arange(
                    len(targets),
                    dtype=np.int64
                )

            total_pairs += len(nearby_ids)
            candidate_nearby.append( nearby_ids )

        all_ids = np.unique( np.concatenate(candidate_nearby) )
        print(
            f"Union targets: "
            f"{len(all_ids)}"
        )

        if len(all_ids) == 0: continue
        inverse = np.full( len(targets), -1, dtype=np.int32 ); inverse[all_ids] = np.arange( len(all_ids), dtype=np.int32 )

        t0 = time.time()
        pred = predict_error(
            model,
            batch_cache,
            target_cache,
            all_ids,
            metric
        )

        t_predict += ( time.time() - t0 )
        for local_idx, nearby_ids in enumerate(
            candidate_nearby
        ):

            cols = inverse[nearby_ids]

            local_pred = pred[
                local_idx,
                cols
            ]

            base = target_cache["error"][nearby_ids]
            inc = np.maximum(base - local_pred, 0)

            gain[start_idx + local_idx] = inc.sum()

            improved[start_idx + local_idx] = (
                inc > IMPROVEMENT_THRESHOLD
            ).sum()

        


        #gain[c0:c0+len(cand)],improved[c0:c0+len(cand)]=bg,bi
        done=end_idx; elapsed=time.time()-start
        eta = ( elapsed / done * ( len(candidate_cache["x"]) - done ) )
        print(
            f"Average nearby targets: "
            f"{total_pairs / len(candidate_nearby):.0f}"
        )
        print(f"Total prediction time: {t_predict:.2f} s")
        print(f"Candidates {done}/{len(candidate_cache['x'])}; elapsed {elapsed/60:.1f} min; ETA {eta/60:.1f} min")
    return gain,improved




def run(name, years, metric="prediction_rmse", force=False, grid_file=None, output_dir=None,cell_inputs=False):
    p = paths( name, years, metric, output_dir=output_dir )
    report_progress("Leser terreng og observasjoner")
    full=load_grid(grid_file)
    from grid_resolution import training_spacing
    cell_m = training_spacing(grid_file) if cell_inputs else 1000
    stations=station_features(full,p,force,cell_inputs=cell_inputs)

    print(f"Stations before Norway filter: {len(stations)}")
    norway = pyogrio.read_dataframe(NORWAY_POLYGON_FILE)
    norway_geom = norway.geometry.iloc[0]
    prepared_norway = prep(norway_geom)
    original_positions = (gpd.GeoSeries(gpd.points_from_xy(stations.lon, stations.lat), crs=4326).to_crs(25833)
                          if cell_inputs else stations.geometry)
    stations["inside_norway"] = [
        prepared_norway.contains(g)
        for g in original_positions
    ]
    stations = stations.loc[stations["inside_norway"]]
    print(f"Stations after Norway filter: {len(stations)}")

    previous_metadata = p["metadata"].read_text() if p["metadata"].exists() else None
    pairs, stations = build_pairs(name, years, stations, p, metric, force,cell_inputs=cell_inputs,cell_m=cell_m)
    inputs_changed = (previous_metadata != p["metadata"].read_text() or
                      (p["model"].exists() and p["model"].stat().st_mtime_ns < p["metadata"].stat().st_mtime_ns))
    report_progress("Tilpasser modell (tidsanslag ikke tilgjengelig ennå)")
    fit_error_model(pairs, p, force=force or inputs_changed)
    print(f"Training complete: {len(stations)} stations; {p['model'].name}")