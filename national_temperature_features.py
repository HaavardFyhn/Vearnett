"""Inference-time features shared by national training preparation and maps."""
import numpy as np

from train_valley_pilot import static_features


def temperature_features(table, source, targets, graph, temperature, stamp, fields, projection_xy):
    targets = np.asarray(targets, dtype=int)
    projection_xy = np.asarray(projection_xy, dtype=np.float64)
    a, b = fields[source], fields[targets]
    vectors = projection_xy[targets]-projection_xy[source]
    lengths = np.linalg.norm(vectors, axis=1)
    bearing = np.divide(vectors, lengths[:, None], out=np.zeros_like(vectors),
                        where=lengths[:, None] > 0)
    along = np.sum(b[:, 1:3]*bearing, axis=1)
    speed_a = np.linalg.norm(a[1:3])
    speed_b = np.linalg.norm(b[:, 1:3], axis=1)
    delta = b[:, 0]-a[0]
    calendar = [np.sin(2*np.pi*stamp.dayofyear/365.25),
                np.cos(2*np.pi*stamp.dayofyear/365.25), stamp.hour/6]
    rows = [
        [temperature, *calendar, *static_features(table, source, int(target)),
         *a, *b[i], delta[i], temperature-a[0], speed_a, speed_b[i],
         *graph[i], along[i], abs(along[i])*graph[i, 2],
         speed_a*graph[i, 1], speed_b[i]*graph[i, 2],
         (1-b[i, 3])*graph[i, 2], (temperature-a[0])*graph[i, 1]]
        for i, target in enumerate(targets)]
    return np.asarray(rows, np.float32).reshape(-1, 39), delta.astype(np.float32)
