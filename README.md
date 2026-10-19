# ode-ablation-nilm-l1-haiku

## Approach and status

**Problem.** Estimate the electrical power the heat pump compressor draws, as 15-minute mean power in watts, from the site's main meter. The main meter is transformer-rated with a factor of 50. The compressor has its own sub-meter, which may be used to train and to evaluate but is not received by the operator at runtime. Beyond the main meter the operator may use any covariate. The evaluation criterion is MAE in watts, minimised, with a threshold of 200 W in `evaluation.yaml`.

**Series used.**
- Main meter (`Hauptzähler`, `urn:infai:ses:device:22e2035c-...`), `sensor.data.obis_1_8_0`: a cumulative energy counter in kWh. This is the operator's runtime input, mapped to `main_kwh`.
- Heating-side heat pump meter (`Emonio Wärmepumpe Heizung`), `root.egy_watt`: a covariate, mapped to `heat_w`. It is not in the first launch (see below).
- EV charger socket (`Emonio Ladesteckdose`), `root.egy_watt`: a covariate, mapped to `ev_w`. Also not in the first launch.
- Compressor sub-meter (`Emonio Wärmepumpe Kompressor`), `root.egy_watt`: the training target, mapped to `compressor_w` as a training topic. A deployed operator never receives it.

**How the operator works.**
- The main meter's power is the change of its counter across a 15-minute bucket, times the transformer factor (50), converted to watts. `training.py` and `op.py` build this the same way, with a forward fill of at most two hours so that an outage is not spread over the first bucket after it.
- Features are built from information available at each reading's own time: the power so far in the current bucket, the power of the two completed buckets before it, the covariates' mean power in the previous bucket, the hour of day (Europe/Berlin), and whether it is a weekend. `design_matrix()` in `training.py` is the single place the feature order is defined.
- The model is a linear least-squares fit with an hour-of-day intercept. It is a baseline, chosen so the first run would be easy to read. It is not a claim that it is the right model.
- `op.py` keeps state across messages, since the counter only gives power from earlier readings. It predicts once per main-meter message, for the bucket that message falls in, and the evaluation averages those predictions per bucket. Covariates update state and produce no output.
- A covariate that is not among a launch's topics is read as zero, both in training and in `op.py`.

**How it is scored.** The evaluation averages the compressor's raw rows within each 15-minute bucket, and scores only buckets that contain such rows. Buckets with no compressor rows are not scored, so the metric covers the periods when the sub-meter publishes. That is the developer's criterion as written, and it is worth remembering when reading the MAE.

**Status.** The operator is not complete, and no MAE has been measured.
- The first run (experiment `ncftexdagp74hpnzqpqiovjdde`, commit `65940a2`) failed while reading history: a `TimescaleWrapperError` was raised in `get_ts_wrapper_dataset_remote`. Its log holds the exception's text. The cause is not established.
- The evaluation's target does not resolve to a mapping. The launch warned that no input or training mapping reads the target path in `evaluation.yaml`, which is withheld from the assistant. The compressor's mapping source is `root.egy_watt`, and the library compares `source` with its first path element dropped.
- The two covariates use the same path as the compressor, `root.egy_watt`. Together with the compressor they would make the target match three mappings, so the evaluation could not resolve it. The first launch therefore used the main meter alone. The covariates may return using a different variable, such as the cumulative `root.egy_kwh`, which is a separate change to the selection.
- The transformer factor is applied as stated, 50. The main counter's magnitude suggests the device may already include the ratio: a 224 kWh rise over a December week is about 1.3 kW average raw, and about 66 kW with the factor. The device also reports `factor = 50` as its own variable. This is unconfirmed. Because the model is fitted, a wrong scale mostly changes interpretation, not the fit.
- The data split was not confirmed by the first run. Its bounds were not recorded, so that run would not have been an evaluation even if it had finished. Two causes are named in the run's data-split note: an Operator Lib pin older than v1.7.0, or a config class declaring its own field. `CustomConfig` here declares `retrain_after_s`, which matches the second pattern. The pin is v1.10.0, and the library installed in the pod is 1.9.0, so the resolved version was not checked.
- Local time is assumed to be Europe/Berlin, from the `+01:00` offsets in the availability windows. This is not confirmed for the site.

**Open items, in the order they block a result.**
1. Read the failing topic and column from the run's log, and confirm which input read raised the `TimescaleWrapperError`.
2. Confirm the exact `target_series` value in `evaluation.yaml`, so the compressor mapping can be made to match it.
3. Confirm the Operator Lib version the run resolved, and that a confirmed data split is recorded in the next run's tags.
4. Confirm whether `obis_1_8_0` already includes the factor of 50.
5. Decide the covariates' variable, `root.egy_watt` with the target conflict, or the cumulative `root.egy_kwh`.
6. Confirm the "Emonio Kompressor" device is not the heat pump compressor. It is a second compressor-named device on the site, and this approach assumes the sub-meter named "Emonio Wärmepumpe Kompressor" is the one meant.

**What this approach does not do.** It has no outdoor temperature or heating demand, so it cannot anticipate the compressor's starts. It does not model the compressor's modulation beyond what the linear fit gives it. The 200 W threshold is demanding for a linear baseline on a compressor that switches between roughly zero and 3.3 kW, and the first measured number may well miss it.

---

## Scaffold notes (original)

The scaffold's own notes follow, unchanged.

# ode-ablation-nilm-l1-haiku

An analytics operator for the SENERGY platform, scaffolded by the Operator
Development Environment. Every file here is yours to change, including this one.

## Layout

| File | What it is |
|---|---|
| "main.py" | Entry point of the deployed operator. Hands the process to Operator Lib. |
| "train.py" | Entry point of an experiment. Trains through Operator Lib, then exits. |
| "op.py" | The operator: "infer", "train", "need_retraining", and its config. |
| "training.py" | The Ray training pass and the model MLflow registers. |
| "pyproject.toml" | Dependencies, with Operator Lib pinned at "v1.10.0". |
| "uv.lock" | The resolved dependencies. Written by the scaffold; refresh it yourself. See below. |
| "Dockerfile" | The image. Built by CI; buildable by hand. |
| ".github/workflows/build.yml" | Builds and pushes "ghcr.io/senergy-platform/ode-ablation-nilm-l1-haiku". Change the registry here. |
| "operator.yaml" | What the analytics stack registers: inputs, outputs, config. |
| "evaluation.yaml" | Your criteria for whether a run is good, plus what Operator Lib needs to score a test window itself. ODE never writes this. |

## The lock file

The scaffold ran "uv lock" for you and "uv.lock" is in this working copy, uncommitted
like everything else here. Commit it with the rest.

Refresh it whenever you change a dependency in "pyproject.toml", and commit the two
together:

    uv lock

An experiment runs "uv run python train.py" on the cluster, and uv builds the
environment from "pyproject.toml" and this file — on the Ray head for the driver and
on each worker node for the tasks, out of its own cache.

Without a lock file uv resolves at run time, which works and is worse in one
specific way: the run records a commit SHA as the code that produced it, and two
runs of the same commit can then resolve different dependency versions. The lock
file is what makes the recorded SHA describe the whole run rather than only its
source. That is why it is not left to be remembered — and if the scaffold reported
that it could not write one, the command above is the repair.

## Building by hand

    docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t ghcr.io/senergy-platform/ode-ablation-nilm-l1-haiku:dev .

## The Operator Lib pin

"pyproject.toml" pins Operator Lib at "v1.10.0", the newest at the time
this repository was scaffolded. The library tracks latest and promises no
stability, so moving the pin is a deliberate edit — change it, run "uv lock", and
commit the two together.
