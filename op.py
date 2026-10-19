"""The operator: what it infers per message, and when it retrains.

MLOperator is the machine-learning half of Operator Lib. It loads the model
registered under this pipeline and operator from MLflow, calls infer() for every
message that matches a selector, and calls train() when there is no model yet or
when need_retraining() says so. Training runs on Ray.

infer() keeps state across messages: the main meter is a cumulative counter, so
its power is only known from the readings before it. The state is rebuilt by the
replay during an evaluation, and by the message stream when deployed.

Three methods are yours. Everything else belongs to the library.
"""

import datetime
import typing

import numpy as np
from mlflow.pyfunc import PyFuncModel, PythonModel

from operator_lib.util import Config, MLOperator, Selector
from operator_lib.util.helpers import TrainMlflowLogger

from training import (
    BUCKET_S,
    EV,
    FFILL_BUCKETS,
    HEAT,
    LOCAL_TZ,
    MAIN,
    METER_FACTOR,
    MIN_ELAPSED_S,
    train_model,
)

# Buckets of history kept in memory: the two lags, the forward fill, and a margin.
KEEP_BUCKETS = 32


class CustomConfig(Config):
    """Deployment configuration, typed.

    The base Config already carries mlflow_url, ray_url and ts_conn. Anything
    added here arrives from the operator's deployment config under the same name,
    with this value as the default.
    """

    # Retrain at most this often, in seconds. A day, so a deployment does not
    # spend its life training.
    retrain_after_s = 86400


class Operator(MLOperator):
    configType = CustomConfig

    # The compressor's own sub-meter is not an input: it is a training topic, which
    # a deployed operator never receives. The main meter is the one permitted
    # signal; the two other inputs are covariates.
    selectors = [
        Selector({"name": "main", "args": [MAIN]}),
        Selector({"name": "heat", "args": [HEAT]}),
        Selector({"name": "ev", "args": [EV]}),
    ]

    def init(self, *args, **kwargs):
        # State first: under a data split, super().init() trains and replays the
        # test window before it returns, so anything set after it is missing
        # during the replay.
        self.trained_at: typing.Optional[datetime.datetime] = None
        self._energy: typing.Dict[int, float] = {}
        self._covariates: typing.Dict[str, typing.Dict[int, typing.List[float]]] = {"heat": {}, "ev": {}}
        self._newest = 0
        super().init(*args, **kwargs)

    def infer(
        self,
        model: typing.Optional[PyFuncModel],
        data: typing.Dict[str, typing.Any],
        selector: str,
        device_id: str,
        timestamp: datetime.datetime,
    ) -> typing.Tuple[
        typing.Optional[datetime.datetime], typing.Optional[typing.Any], typing.Optional[PythonModel]
    ]:
        """Called for every message. Returns (result timestamp, result, new model).

        Only a main-meter message yields a prediction, for the 15-minute bucket it
        falls in. Covariate messages only update state.
        """
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)
        seconds = timestamp.timestamp()
        bucket = int(seconds // BUCKET_S)
        self._newest = max(self._newest, bucket)

        if selector == "main":
            value = data.get(MAIN)
            if value is None:
                return None, None, None
            self._energy[bucket] = float(value)
            self._prune()
            if model is None:
                return None, None, None
            features = self._features(float(value), bucket, seconds - bucket * BUCKET_S, timestamp)
            if features is None:
                return None, None, None
            result = model.predict(features)
            return None, {"prediction": float(np.ravel(result)[0])}, None

        if selector in self._covariates:
            value = data.get(HEAT if selector == "heat" else EV)
            if value is not None:
                cell = self._covariates[selector].setdefault(bucket, [0.0, 0])
                cell[0] += float(value)
                cell[1] += 1
            self._prune()
        return None, None, None

    def train(
        self, model: typing.Optional[PyFuncModel], logger: TrainMlflowLogger
    ) -> typing.Optional[PythonModel]:
        """Called when there is no model, or when need_retraining() said so.

        Runs inside a Ray session the library opened, with an MLflow run already
        started, so params and metrics logged through "logger" land on the run the
        resulting model is registered from.
        """
        self.trained_at = datetime.datetime.now(datetime.timezone.utc)
        return train_model(logger)

    def need_retraining(self, model: typing.Optional[PyFuncModel]) -> bool:
        """Called after every inference, so it has to be cheap.

        Time-based to start with. A better policy watches the data, a drift in the
        input distribution or an error that stops falling, and that is a decision to
        make with the profile in front of you rather than a default to inherit.
        """
        if model is None:
            return True
        if self.trained_at is None:
            return False
        age = datetime.datetime.now(datetime.timezone.utc) - self.trained_at
        return age.total_seconds() >= self.config.retrain_after_s

    # -- state, matching training.py's definitions bucket for bucket --

    def _last_energy(self, k: int) -> typing.Optional[float]:
        """The last cumulative reading in bucket k, or carried forward from up to
        FFILL_BUCKETS earlier buckets, as training's forward fill does."""
        for j in range(k, k - FFILL_BUCKETS - 1, -1):
            if j in self._energy:
                return self._energy[j]
        return None

    def _power(self, k: int) -> typing.Optional[float]:
        """Mean power of bucket k in watts: the change of the counter across it."""
        now = self._last_energy(k)
        before = self._last_energy(k - 1)
        if now is None or before is None:
            return None
        return (now - before) * METER_FACTOR * 4000.0

    def _mean(self, name: str, k: int) -> float:
        cell = self._covariates[name].get(k)
        if not cell or cell[1] == 0:
            return 0.0
        return cell[0] / cell[1]

    def _features(
        self, value: float, bucket: int, elapsed: float, timestamp: datetime.datetime
    ) -> typing.Optional[typing.Dict[str, typing.Any]]:
        p1 = self._power(bucket - 1)
        p2 = self._power(bucket - 2)
        if p1 is None or p2 is None:
            return None
        if elapsed >= MIN_ELAPSED_S:
            before = self._last_energy(bucket - 1)
            if before is None:
                return None
            p0 = (value - before) * METER_FACTOR * 3.6e6 / max(elapsed, 1.0)
        else:
            p0 = p1
        local = timestamp.astimezone(LOCAL_TZ)
        return {
            "hour": local.hour,
            "weekend": int(local.weekday() >= 5),
            "p0": p0,
            "p1": p1,
            "p2": p2,
            "heat1": self._mean("heat", bucket - 1),
            "ev1": self._mean("ev", bucket - 1),
        }

    def _prune(self) -> None:
        cutoff = self._newest - KEEP_BUCKETS
        self._energy = {k: v for k, v in self._energy.items() if k >= cutoff}
        for name, cells in list(self._covariates.items()):
            self._covariates[name] = {k: v for k, v in cells.items() if k >= cutoff}
