# OpenRadioss Parametric Rear-Impact Pipeline

A Python pipeline for parametric rear-impact crash simulations using the VIVA+ Human Body Model (HBM) in OpenRadioss. It automates deck modification, batch job submission, metric extraction, and HTML reporting across a user-defined sweep of cervical muscle activation conditions.

## Requirements

- Python 3.9+
- OpenRadioss installed and on `PATH` (`starter_linux64_gf` and `engine_linux64_gf` must be executable)
- VIVA+ HBM deck in `.rad` format
- `neck_muscles.csv` mapping part names to muscle identifiers

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Validate deck parsing and muscle group mapping
python3 run_sweep.py validate --deck path/to/viva_plus.rad --csv neck_muscles.csv

# 2. Dry-run: preview the sweep plan without submitting any jobs
python3 run_sweep.py sweep --deck path/to/viva_plus.rad --csv neck_muscles.csv --config sweep_config_example.json --output results/ --dry-run

# 3. Run the full sweep
python3 run_sweep.py sweep --deck path/to/viva_plus.rad --csv neck_muscles.csv --config sweep_config_example.json --output results/
```

A third subcommand re-runs post-processing on a completed output directory without re-running simulations:

```bash
python3 run_sweep.py report --results-dir results/ --output reports/
```

## Sweep Config

`sweep_config_example.json` is a JSON array where each object defines one simulation run. Keys match `RunConfig` dataclass fields exactly.

| Field | Type | Required | Default | Valid Range / Notes |
|---|---|---|---|---|
| `run_id` | `str` | yes | - | Unique label; used as output subdirectory name (`run_{run_id}/`) |
| `activated_groups` | `list[str]` | yes | - | Subset of 8 cervical group tokens (see Muscle Groups); `[]` = passive baseline |
| `t_start` | `float` | yes | - | Activation onset time in seconds; negative = pre-impact |
| `magnitude_pct` | `float` | yes | - | Peak activation as % MVC; 0 - 150 |
| `baseline_tone_pct` | `float` | no | `3.5` | Resting tone for non-activated groups as % MVC; 0 - 10 |
| `peak_g` | `float` | no | `10.0` | Crash pulse peak acceleration in g; 5 - 15 |
| `notes` | `str` | no | `""` | Free-text annotation injected into the deck comment header |

Example entry:

```json
[
  {
    "run_id": "scm_early_80pct",
    "activated_groups": ["SCM"],
    "t_start": -0.020,
    "magnitude_pct": 80.0,
    "baseline_tone_pct": 2.0,
    "peak_g": 8.0,
    "notes": "SCM bilateral pre-activation at 80% MVC, T_start=-20ms, 8g."
  }
]
```

See `sweep_config_example.json` for a full 8-run reference design covering passive baseline, isolated muscle groups, full co-contraction, and supramaximal worst-case conditions.

## Muscle Groups

`parametric_editor.py` maps VIVA+ cervical muscle parts into 8 functional groups via the `ALL_CERVICAL_GROUPS` list. Use these exact token strings in `activated_groups`.

| Token | Anatomical substrings matched |
|---|---|
| `SCM` | Sternocleidomastoid, SCM |
| `STH` | Sternohyoid, Sternothyroid, Omohyoid |
| `Scal` | Scalenus-Anterior, Scalenus-Medius, Scalenus-Posterior |
| `Trap` | Trapezius |
| `SCap` | Splenius-Capitis |
| `SCerv` | Splenius-Cervicis |
| `CM_C4` | Semispinalis-Capitis, Longissimus-Capitis |
| `CM_C6` | Semispinalis-Cervicis, Longissimus-Cervicis, Multifidus, Longus-Capitis, Longus-Colli |

Matching is case-insensitive against the `muscle_name` column in `neck_muscles.csv` and against VIVA+ `/PART` title strings. Parts that match no group are flagged in the validation summary but are never modified.

## Outputs

All outputs are written under the `--output` directory.

### `sweep_report.csv`

One row per simulation run. Key columns:

| Column | Description |
|---|---|
| `run_id` | Run identifier from `RunConfig` |
| `activated_groups` | Semicolon-separated list of activated group tokens |
| `t_start_ms` | Activation onset converted to milliseconds |
| `magnitude_pct` | Peak MVC % used |
| `peak_g` | Crash pulse peak g |
| `HIC15` | Head Injury Criterion (15 ms window) |
| `peak_head_accel_g` | Peak resultant head CG acceleration (g) |
| `peak_head_t1_angle_deg` | Peak head-to-T1 relative angle (degrees) |
| `upper_neck_Fx_N` | Peak upper neck anterior shear force (N) |
| `upper_neck_Fz_N` | Peak upper neck axial compression force (N) |
| `upper_neck_My_Nm` | Peak upper neck flexion/extension moment (Nm) |
| `lower_neck_Fx_N` | Peak lower neck anterior shear force (N) |
| `lower_neck_Fz_N` | Peak lower neck axial compression force (N) |
| `lower_neck_My_Nm` | Peak lower neck flexion/extension moment (Nm) |

Failed runs appear as rows with `NaN` metric values.

### `sweep_report.html`

Self-contained HTML report (no external dependencies) with a sortable results table, distribution plots, and per-run status summary.

### Per-run subdirectories

`results/run_{run_id}/` contains the modified `.rad` deck, OpenRadioss stdout/stderr logs, the T01 time-history output file, and a `run_config.json` sidecar used by the `report` subcommand.

## Notes

- **MVC% range:** 0-150% (supramaximal values up to 150% supported for edge-case testing)
- **T_start:** -0.060s to +0.100s (negative = pre-impact)
- **Hill-type curve:** native VIVA+ activation curve shape is preserved for activated groups; only the time-axis shift (Ashiftx) and scale/shift (Fscaley, Fshifty) are modified
- **Non-cervical muscles:** never modified; only parts in the PID ranges 205000-209999 and 255000-259999 are touched
- **Baseline tone:** non-activated cervical groups receive a flat constant activation equal to `baseline_tone_pct` (default 3.5% MVC); set to `0.0` for a fully passive condition
- **Parallel jobs:** use `--parallel N` on the `sweep` subcommand to run N simultaneous OpenRadioss jobs
- **Crash pulse scaling:** `peak_g` rescales the `/IMPVEL` or `/IMPACC` function amplitude; the pulse shape is preserved

## Files

| File | Purpose |
|---|---|
| `run_sweep.py` | CLI entry point (`validate`, `sweep`, `report` subcommands) |
| `parametric_editor.py` | `RunConfig` dataclass, `ParametricEditor`, `MuscleMapper`, `ALL_CERVICAL_GROUPS` |
| `deck_parser.py` | Block-level `.rad` file parser and writer |
| `muscle_mapper.py` | Standalone muscle group mapping utilities |
| `batch_orchestrator.py` | Parallel job submission and status tracking |
| `postprocessor.py` | T01 time-history parsing, metric extraction, `SweepReporter` |
| `sweep_config_example.json` | Reference sweep config with 8 representative runs |
| `requirements.txt` | Python dependencies |
