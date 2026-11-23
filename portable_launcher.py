"""Entry point for the self-contained Windows application."""
import socket
import sys
import threading
import time
import webbrowser

from station_app import main


def open_when_ready(port):
    for attempt in range(1200):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                webbrowser.open(f"http://127.0.0.1:{port}/")
                return
        except OSError:
            time.sleep(0.5)


def smoke_test():
    import json
    import numpy as np
    from station_app import StationEngine, create_app, default_dataset, OUT

    spacing = json.loads((OUT / "portable_weather.json").read_text(encoding="utf-8"))["cell_m"]
    engine = StationEngine(OUT / "unused.npy", cell_m=spacing)
    try:
        client = create_app(engine, initial_dataset=default_dataset(engine, "temperature")).test_client()
        for path in ("/", "/station_app.js", "/api/state", "/api/counties",
                     "/api/map.png?mode=uncertainty&size=400",
                     "/api/map.bin?mode=uncertainty&size=400", "/api/roads/options"):
            response = client.get(path)
            if response.status_code != 200:
                raise RuntimeError(f"Smoke test failed: {path}: {response.status_code}")
            if path == "/api/roads/options":
                from station_roads import ROAD_LAYER
                if not response.json.get("available") or response.json.get("layer") != ROAD_LAYER:
                    raise RuntimeError(f"Portable road network must use {ROAD_LAYER}")
            print(f"PASS {path} ({len(response.data)} bytes)", flush=True)
        for config in engine.weather.available:
            profile = engine.weather.profile(config["id"])
            if profile["baseline"] is None or not np.isfinite(profile["baseline"]).any():
                raise RuntimeError(f"No usable baseline: {config['id']}")
            print(f"PASS model {config['id']}", flush=True)
        point = engine.grid.iloc[len(engine.grid) // 2]
        lon, lat = engine.to_lonlat.transform(point.x, point.y)
        client.post("/api/action", json=dict(action="network", value="empty"))
        added = client.post("/api/action", json=dict(action="add", lon=lon, lat=lat, sensors=["temperature"]))
        if added.status_code != 200 or len(added.json["stations"]) != 1:
            raise RuntimeError("Portable station prediction failed")
        print("PASS interactive station prediction", flush=True)
    finally:
        engine.resources.close()


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        smoke_test()
    else:
        import json
        from station_app import OUT
        manifest = json.loads((OUT / "portable_weather.json").read_text(encoding="utf-8"))
        sys.argv.extend(["--cell-m", str(manifest["cell_m"])])
        for port in range(8765, 8785):
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) != 0:
                    break
        sys.argv.extend(["--port", str(port)])
        threading.Thread(target=open_when_ready, args=(port,), daemon=True).start()
        main()