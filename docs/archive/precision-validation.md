# Historical v3 image acceptance workflow

Archived only; the current application benchmark has no campaign, quality tier or quality-gated performance entry point.

## v3 image acceptance

The ranked precision tools support an explicit v3 campaign independently of the
legacy image/video validators. Select `--policy-version 3 --quality-tier
balanced` (or `high-fidelity` / `compact`) on `export_precision_outputs.py` and
`validate_precision_regression.py run`. The evaluator reads the pinned policy
from the run; mixed v2/v3 runs and changed tier identities are rejected. Dataset,
normalized-input and ranked tensor payload formats remain version 2; v3 recipes,
campaigns and decisions have their own version and gate hash.

Quality reports expose `task_quality_status`, separate absolute/cache-incremental
results, and `diagnostics`. `qualification_status` remains `NOT_RUN` without
complete independent arithmetic and regression evidence; a known quality failure
is `FAIL`. Passing quality alone never certifies the whole model. Performance
reports separately expose `measurement_status`, `non_regression_status` and
`benefit_status` for each workload. `NOT_DEMONSTRATED` means no benefit label was
earned, and does not mean task quality failed.

Freeze a new campaign before final inference. Declare every prior precision
dataset and its SHA-256 in `evaluation_history`; the freezer excludes previously
used IDs/content and does not open old reserve images. This is a completeness
declaration, not automatic discovery of experiments outside the supplied history.
Use old evaluation images only for development, retaining all original receipts.
The freezer requires passing development checks except for the final image-count
requirement. Quality-only v3 campaigns may omit performance cases/baselines;
compressed-cache candidates still require an exact `cache_baseline`.

See [quality tiers](quantization-qualification.md#v3-quality-tiers-and-optional-benefits) and the
[v3 plan](../plans/20261010-101413-precision-acceptance-v3.md) for the complete contract,
selection-file example, commands and verification scope. The Linux precision
runners currently cover CPU/CUDA execution, not a new Metal integration or video
qualification. Do not use changed quality thresholds to waive implementation
arithmetic failures.
