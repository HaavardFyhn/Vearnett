#!/bin/sh
set -eu

code_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=$(dirname -- "$code_dir")
python="$root/.venv/bin/python"
data="$root/outputs/portable_build/Remote_500m/data"

if [ ! -x "$python" ] || [ ! -f "$data/outputs/portable_weather.json" ]; then
    printf '%s\n' "Run this script from a code folder directly inside ~/vaernett_500m." >&2
    exit 1
fi

trap 'status=$?; printf "\nBatch exit status: %s\n" "$status"' 0

printf '\n%s Starting EMPTY network (assumed reference RMSE: 0.5 degrees C)\n' "$(date -Iseconds)"
"$python" -u "$code_dir/placement_gain.py" \
    --data-root "$data" \
    --work-dir "$root/outputs/placement_road200_radius100_empty_ref05" \
    --station-network empty --empty-reference-rmse 0.5 \
    --radius-km 100 --workers 14 --overwrite

printf '\n%s Empty network finished; starting ALL stations\n' "$(date -Iseconds)"
"$python" -u "$code_dir/placement_gain.py" \
    --data-root "$data" \
    --work-dir "$root/outputs/placement_road200_radius100_all" \
    --station-network all \
    --radius-km 100 --workers 14 --overwrite

printf '\n%s Both placement maps completed\n' "$(date -Iseconds)"
