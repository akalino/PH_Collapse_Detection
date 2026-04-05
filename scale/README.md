
## Step 1: Environment


- pinned Python environment file,
- record package versions,
- verify all PH/TDA dependencies install cleanly,
- document the tested Python version.

---

## Step 2: Containerize

Package the repo into a Docker container so the same code and environment can run locally, on a VM, or in a batch system.

- add a `Dockerfile`,
- install all required dependencies,
- make the pipeline runnable from the container,
- support running either the full pipeline or individual stages.

---

## Step 3: Config-driven experiment settings


- move $n, d, \varepsilon$, worker count, simulation count, and output paths out of the source code,
- pass these through config file,
- keep the current small values as defaults for local testing.

---

## Step 4: Make outputs run-specific

- give every run a unique `run_id`,
- write all outputs into a dedicated run directory,
- stop using shared output folders that are cleared between runs,
- keep artifacts, logs, and summary files grouped by run.

---

## Step 5: Fix reproducibility and metadata

- replace unstable seed generation with deterministic seeds,
- save metadata for each run,
- log the parameter grid, seed policy, code version, and runtime environment,
- write structured logs instead of relying only on console output.

---

## Step 6: Break the pipeline into stages

- define stages such as calibration, null simulation, alternative simulation, comparison, baselines, and summarization,
- allow each stage to be launched independently,
- make downstream stages check that required inputs exist.

---

## Step 7: Add Cloud Storage support

- create a GCS bucket for experiment outputs,
- organize results by run,
- sync outputs to GCS after each major stage,
- keep configs and metadata in the same bucket layout.

---

## Step 8: Run the first pilot on Compute Engine

- choose one VM with enough RAM for PH computations,
- run a pilot scale-up experiment,
- sync results to Cloud Storage,
- measure runtime and memory usage.

---

## Step 9: Expand to larger grids

- increase the grids for \(n\) and \(d\),
- rerun calibration on the larger null families,
- rerun power experiments on the larger alternative families,
- collect scaling results and updated mechanism maps.

---

## Step 10: Move to Google Cloud Batch

- run the containerized stages as batch jobs,
- submit one job per stage or per experiment block,
- use Cloud Storage as the shared source of truth,
- keep outputs and metadata organized by run.
