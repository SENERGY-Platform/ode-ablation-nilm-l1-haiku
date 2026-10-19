"""Training, on Ray.

The operator estimates the power the heat pump compressor draws from the site's
main meter. It is trained against the compressor's own sub-meter, which is a
training topic: a deployed operator never receives it.

The main meter publishes a cumulative energy counter. Its power over a 15-minute
bucket is the change of that counter across the bucket, times the transformer
factor (METER_FACTOR), converted to watts. Features are built the same way here
and in op.py; design_matrix() is the single place the feature order is defined.

A covariate that is not among the launch's input topics is read as zero, both
here and in op.py, so a launch with the main meter alone is consistent.

The model is a linear least-squares fit with an hour-of-day intercept. It is a
baseline for this operator, not a claim that it is the right model.
"""

import datetime
import typing
import zoneinfo

import numpy as np
import pandas as pd
import ray
from mlflow.pyfunc import PythonModel

from operator_lib.util.helpers import provide_historic_data, TrainMlflowLogger


TRAINING_WINDOW = datetime.timedelta(days=365)

# Transformer ratio of the main meter, as the developer stated it. The counter is
# multiplied by this. Whether the device counter already includes the ratio is not
# established; see the note in the hand-over.
METER_FACTOR = 50.0

LOCAL_TZ = zoneinfo.ZoneInfo("Europe/Berlin")
BUCKET = "15min"
BUCKET_S = 900
# A cumulative reading is carried forward for at most this many buckets (two
# hours), so a long outage is not spread over the first bucket after it.
FFILL_BUCKETS = 8
# Before this many seconds of a bucket have passed, the completed previous
# bucket's power stands in for the partial one.
MIN_ELAPSED_S = 120

# Destination names, as the launch maps them. Keep in step with op.py and
# operator.yaml.
MAIN = "main_kwh"
HEAT = "heat_w"
EV = "ev_w"
TARGET = "compressor_w"

N_FEATURES = 30  # 24 hour dummies, weekend, p0, p1, p2, heat1, ev1


def design_matrix(hour, weekend, p0, p1, p2, heat1, ev1) -> np.ndarray:
    """The feature matrix, one row per observation, in the one fixed order.

    p0 is the power so far in the current bucket, p1 and p2 the power of the
    two completed buckets before it, heat1 and ev1 the mean power of the two
    covariates in the previous bucket.
    """
    arrays = np.broadcast_arrays(
        np.asarray(hour, dtype=int),
        np.asarray(weekend, dtype=float),
        np.asarray(p0, dtype=float),
        np.asarray(p1, dtype=float),
        np.asarray(p2, dtype=float),
        np.asarray(heat1, dtype=float),
        np.asarray(ev1, dtype=float),
    )
    hour, weekend, p0, p1, p2, heat1, ev1 = [a.reshape(-1) for a in arrays]
    x = np.zeros((hour.size, N_FEATURES))
    x[np.arange(hour.size), hour] = 1.0
    x[:, 24] = weekend
    x[:, 25] = p0
    x[:, 26] = p1
    x[:, 27] = p2
    x[:, 28] = heat1
    x[:, 29] = ev1
    return x


class OdeAblationNilmL1HaikuModel(PythonModel):
    """The model MLflow registers and op.py loads.

    predict() receives exactly the feature payload op.py builds, so the two are
    one contract in two files. Keep it serialisable: MLflow stores this object.
    """

    def __init__(self, coefficients: typing.List[float]) -> None:
        self.coefficients = [float(c) for c in coefficients]

    def predict(self, context, model_input=None, params=None):
        payload = model_input if model_input is not None else context
        x = design_matrix(
            payload["hour"],
            payload["weekend"],
            payload["p0"],
            payload["p1"],
            payload["p2"],
            payload["heat1"],
            payload["ev1"],
        )
        value = float(x[0] @ np.asarray(self.coefficients))
        return max(0.0, value)


def _series(refs, dest: str) -> pd.Series:
    """The values of one destination column, indexed by UTC time, from whichever
    dataset carries it. Rows without a value are dropped."""
    for ref in refs:
        ds = ray.get(ref) if isinstance(ref, ray.ObjectRef) else ref
        parts = []
        for batch in ds.iter_batches(batch_size=65536, batch_format="pandas"):
            if dest in batch.columns and "time" in batch.columns:
                part = pd.DataFrame(
                    {
                        "time": pd.to_datetime(batch["time"], utc=True),
                        "value": pd.to_numeric(batch[dest], errors="coerce"),
                    }
                )
                parts.append(part.dropna())
        if parts:
            frame = pd.concat(parts)
            return frame.set_index("time")["value"].sort_index()
    raise ValueError(f"no topic carries {dest!r}")


def _training_rows(main: pd.Series, heat: pd.Series, ev: pd.Series, target: pd.Series) -> pd.DataFrame:
    """One row per main-meter reading, labelled with the compressor's 15-minute
    mean for the bucket that reading falls in. Only features available at the
    reading's own time are used."""
    last = main.resample(BUCKET).last().ffill(limit=FFILL_BUCKETS)
    power = (last - last.shift(1)) * METER_FACTOR * 4000.0
    heat_b = heat.resample(BUCKET).mean().fillna(0.0)
    ev_b = ev.resample(BUCKET).mean().fillna(0.0)
    label = target.resample(BUCKET).mean()

    rows = pd.DataFrame({"energy": main.to_numpy()}, index=main.index)
    rows["bucket"] = rows.index.floor(BUCKET)
    bucket_index = pd.DatetimeIndex(rows["bucket"])
    prev = bucket_index - pd.Timedelta(BUCKET)
    prev2 = prev - pd.Timedelta(BUCKET)

    le_prev = last.reindex(prev).to_numpy()
    p1 = power.reindex(prev).to_numpy()
    p2 = power.reindex(prev2).to_numpy()
    elapsed = np.asarray((rows.index - bucket_index).total_seconds(), dtype=float)
    partial = (rows["energy"].to_numpy() - le_prev) * METER_FACTOR * 3.6e6 / np.maximum(elapsed, 1.0)
    rows["p0"] = np.where(elapsed >= MIN_ELAPSED_S, partial, p1)
    rows["p1"] = p1
    rows["p2"] = p2
    rows["heat1"] = heat_b.reindex(prev).fillna(0.0).to_numpy()
    rows["ev1"] = ev_b.reindex(prev).fillna(0.0).to_numpy()
    rows["label"] = label.reindex(bucket_index).to_numpy()
    return rows.dropna(subset=["label", "p0", "p1", "p2"])


def train_model(logger: TrainMlflowLogger) -> typing.Optional[PythonModel]:
    """Read the history, fit, and hand back a model for MLflow to register."""
    with logger.trace("read history"):
        inputs = provide_historic_data(TRAINING_WINDOW)
        labels = provide_historic_data(TRAINING_WINDOW, role="training")
        if not inputs or not labels:
            return None
        main = _series(inputs, MAIN)
        try:
            heat = _series(inputs, HEAT)
        except ValueError:
            # Not among this launch's topics: the operator reads it as zero too.
            heat = pd.Series(0.0, index=main.index)
        try:
            ev = _series(inputs, EV)
        except ValueError:
            ev = pd.Series(0.0, index=main.index)
        target = _series(labels, TARGET)

    rows = _training_rows(main, heat, ev, target)
    if rows.empty:
        # Explicitly nothing rather than a model fitted on no data: returning None
        # leaves the previously registered model in place.
        return None

    with logger.trace("fit"):
        local = rows.index.tz_convert(LOCAL_TZ)
        X = design_matrix(
            np.asarray(local.hour),
            np.asarray(local.dayofweek >= 5, dtype=float),
            rows["p0"].to_numpy(),
            rows["p1"].to_numpy(),
            rows["p2"].to_numpy(),
            rows["heat1"].to_numpy(),
            rows["ev1"].to_numpy(),
        )
        y = rows["label"].to_numpy(dtype=float)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        pred = np.maximum(X @ beta, 0.0)
        mae = float(np.mean(np.abs(pred - y)))

    logger.log_params(
        {
            "training_window_days": TRAINING_WINDOW.days,
            "meter_factor": METER_FACTOR,
            "covariates": "none" if HEAT not in _names(inputs) and EV not in _names(inputs) else "partial",
            "model": "linear_hour_of_day_with_partial_bucket",
        }
    )
    logger.log_metrics({"train_rows": float(len(y)), "train_mae_w": mae})
    return OdeAblationNilmL1HaikuModel(coefficients=beta.tolist())


def _names(refs) -> typing.Set[str]:
    """The destination columns the input datasets carry, for the run's params."""
    names = set()
    for ref in refs:
        ds = ray.get(ref) if isinstance(ref, ray.ObjectRef) else ref
        names.update(ds.schema().names)
    return names
