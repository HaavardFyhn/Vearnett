"""Serve Vaernett with Waitress for an approved internal-network test.

Local test: python host_station_app.py
LAN test: python host_station_app.py --host 0.0.0.0 --port 8766
No firewall rules are changed. This launcher provides no authentication or TLS;
use only an IT-approved internal test network, never public port forwarding.
Normal station_app.py startup remains loopback-only and unchanged.
"""
import argparse
import ipaddress
from pathlib import Path
import socket

ROOT = Path(__file__).resolve().parent
PHENOMENA = ("temperature", "wind", "precipitation", "snow", "humidity", "road_temperature")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="Listen address; use 0.0.0.0 explicitly for LAN access")
    parser.add_argument("--port", type=int, default=8766, help="Fixed port; does not silently select another port")
    parser.add_argument("--threads", type=int, default=4, help="Waitress worker threads (1-32)")
    parser.add_argument("--phenomenon", choices=PHENOMENA, default="temperature")
    parser.add_argument("--surfaces", type=Path, default=ROOT / "outputs/uncertainty_surfaces_1000m.npy")
    args = parser.parse_args(argv)
    try:
        ipaddress.ip_address(args.host)
    except ValueError:
        parser.error("--host must be a literal IPv4 or IPv6 address")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not 1 <= args.threads <= 32:
        parser.error("--threads must be between 1 and 32")
    return args


def create_application(args):
    from station_app import StationEngine, KuldehullLayer, OUT, default_dataset, create_app

    print("Loading 1 km terrain, models and saved surfaces ...", flush=True)
    engine = StationEngine(args.surfaces, cell_m=1000, frost_neighbors=0)
    try:
        initial = default_dataset(engine, args.phenomenon)
        engine.warm_weather_maps(dataset=initial)
        engine.warm_placement_maps()
        basins = KuldehullLayer(OUT / "potential_lakes_norway_DTM10_sill20_tiled/potential_lakes.gpkg")
        basins.prepare()
        app = create_app(engine, initial_dataset=initial, kuldehull_layer=basins)
        app.config["SESSION_COOKIE_NAME"] = f"station_lab_hosted_{args.port}"
        return app, engine
    except BaseException:
        engine.resources.close()
        raise


def serve_application(app, args):
    from waitress import serve

    serve(app, host=args.host, port=args.port, threads=args.threads,
          expose_tracebacks=False, clear_untrusted_proxy_headers=True,
          max_request_body_size=4096, connection_limit=32, channel_timeout=120,
          ident="Vaernett")


def main(argv=None):
    args = parse_args(argv)
    try:
        import waitress
    except ImportError:
        raise SystemExit("Install hosting dependencies: python -m pip install -r requirements-hosting.txt") from None
    app, engine = create_application(args)
    try:
        address = ipaddress.ip_address(args.host)
        if address.is_loopback:
            host = f"[{args.host}]" if address.version == 6 else args.host
            print(f"Local test: http://{host}:{args.port}/", flush=True)
        else:
            print(f"Internal test server: http://{socket.getfqdn()}:{args.port}/", flush=True)
            print(f"Listening on {args.host}:{args.port}. Colleagues need an approved network/firewall route.", flush=True)
            print("No login or HTTPS. Do not expose this port to the internet.", flush=True)
        print("Keep this window open; Ctrl+C stops the test server.", flush=True)
        serve_application(app, args)
    except KeyboardInterrupt:
        print("Test server stopped.", flush=True)
    finally:
        engine.resources.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())