"""
run_sweep.py - CLI entry point for OpenRadioss parametric sweep.

Subcommands
-----------
  validate   Validate deck parsing and muscle group mapping.
  sweep      Run a full parametric sweep end-to-end.
  report     Re-run post-processing on existing completed run directories.

Usage
-----
  python run_sweep.py validate --deck model.rad --csv neck_muscles.csv
  python run_sweep.py sweep --deck model.rad --csv neck_muscles.csv \
      --config sweep_config.json --output ./runs [--dry-run] [--parallel 4]
  python run_sweep.py report --results-dir ./runs --output ./reports
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Module-level logger (reconfigured inside each subcommand once output_dir
# is known)
# ---------------------------------------------------------------------------
logger = logging.getLogger("run_sweep")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(output_dir: Optional[Path] = None, level: int = logging.INFO) -> None:
    """Configure structured logging to console + optional file."""
    fmt = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(level)
    # Remove any existing handlers to avoid duplication when called twice
    root.handlers.clear()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # File handler
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "sweep.log"
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)
        logger.info("Logging to %s", log_path)


# ---------------------------------------------------------------------------
# Import project modules (deferred to give meaningful error messages)
# ---------------------------------------------------------------------------

def _import_modules() -> tuple:
    """
    Import all pipeline modules, returning them as a tuple.
    Exits with a clear error if any import fails.
    """
    try:
        # Ensure the tools directory is on sys.path regardless of cwd
        tools_dir = Path(__file__).resolve().parent
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))

        from deck_parser import DeckParser, NeckMuscleCSVParser  # noqa: PLC0415
        from parametric_editor import (  # noqa: PLC0415
            MuscleMapper,
            MUSCLE_GROUP_PATTERNS,
            ALL_CERVICAL_GROUPS,
            RunConfig,
            ParametricEditor,
            load_editor,
        )
        from batch_orchestrator import BatchOrchestrator, JobStatus  # noqa: PLC0415
        from postprocessor import (  # noqa: PLC0415
            THFileParser,
            MetricsExtractor,
            SweepReporter,
            postprocess_sweep,
        )

        return (
            DeckParser,
            NeckMuscleCSVParser,
            MuscleMapper,
            MUSCLE_GROUP_PATTERNS,
            ALL_CERVICAL_GROUPS,
            RunConfig,
            ParametricEditor,
            load_editor,
            BatchOrchestrator,
            JobStatus,
            THFileParser,
            MetricsExtractor,
            SweepReporter,
            postprocess_sweep,
        )
    except ImportError as exc:
        sys.stderr.write(
            f"ERROR: Could not import required module - {exc}\n"
            "Make sure deck_parser.py, parametric_editor.py, "
            "batch_orchestrator.py, and postprocessor.py are on PYTHONPATH "
            "or in the same directory as run_sweep.py.\n"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# RunConfig deserialization from sweep_config.json
# ---------------------------------------------------------------------------

def _load_sweep_config(config_path: str, RunConfig: Any) -> list:
    """
    Load a list of RunConfig objects from a JSON file.

    The JSON file should be an array of objects, each with keys matching
    the RunConfig dataclass fields:
      run_id, activated_groups, t_start, magnitude_pct,
      [baseline_tone_pct], [peak_g], [notes]

    Example
    -------
    [
      {
        "run_id": "baseline",
        "activated_groups": [],
        "t_start": 0.0,
        "magnitude_pct": 0.0
      },
      {
        "run_id": "scm_early",
        "activated_groups": ["SCM"],
        "t_start": -0.020,
        "magnitude_pct": 50.0,
        "peak_g": 8.0
      }
    ]
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Sweep config not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, list):
        raise ValueError(
            f"sweep_config.json must be a JSON array of run config objects, "
            f"got {type(raw).__name__}"
        )

    configs = []
    errors = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            errors.append(f"  Entry {i}: expected dict, got {type(entry).__name__}")
            continue
        try:
            # Keep only fields the RunConfig dataclass declares; extra keys
            # (e.g. postprocessing entity IDs like head_accel_id) are allowed
            # in the config file and consumed later during metric extraction.
            known = {f.name for f in dataclasses.fields(RunConfig)}
            cfg = RunConfig(**{k: v for k, v in entry.items() if k in known})
            configs.append(cfg)
        except (TypeError, ValueError) as exc:
            errors.append(f"  Entry {i} (run_id={entry.get('run_id', '?')}): {exc}")

    if errors:
        raise ValueError(
            f"Errors loading sweep_config.json ({len(errors)} invalid entries):\n"
            + "\n".join(errors)
        )

    logger.info("Loaded %d run configs from %s", len(configs), config_path)
    return configs


# ---------------------------------------------------------------------------
# Validation summary helper
# ---------------------------------------------------------------------------

def _print_validation_summary(mapper: Any, MUSCLE_GROUP_PATTERNS: dict) -> None:
    """Print muscle group mapping counts to stdout."""
    group_counts: dict[str, int] = {g: 0 for g in MUSCLE_GROUP_PATTERNS}

    for pid, group in mapper.part_to_group.items():
        if group in group_counts:
            group_counts[group] += 1

    total_cervical = len(mapper.cervical_pids)
    total_mapped = sum(group_counts.values())
    total_with_funct = len(mapper.part_to_funct)

    print("\n--- Muscle Group Mapping Summary ---")
    print(f"{'Group':<12} {'Parts':>6}")
    print("-" * 20)
    for group, count in group_counts.items():
        flag = "" if count > 0 else "  (no parts found)"
        print(f"{group:<12} {count:>6}{flag}")
    print("-" * 20)
    print(f"{'TOTAL':<12} {total_mapped:>6}")
    print()
    print(f"Total cervical PIDs found  : {total_cervical}")
    print(f"Parts with activation funct: {total_with_funct}")
    print(f"Parts without funct mapping: {total_cervical - total_with_funct}")
    if total_cervical - total_with_funct > 0:
        missing = [
            pid for pid in mapper.cervical_pids
            if pid not in mapper.part_to_funct
        ]
        print(f"  PIDs without funct: {missing[:20]}"
              f"{'...' if len(missing) > 20 else ''}")
    print("--- End Summary ---\n")


# ---------------------------------------------------------------------------
# Console summary table for completed sweep
# ---------------------------------------------------------------------------

def _print_final_table(df: Any) -> None:
    """Print a formatted summary table of sweep results to console."""
    import math

    scalar_cols = [
        "run_id", "activated_groups", "t_start_ms", "magnitude_pct",
        "peak_g", "HIC15", "peak_head_accel_g", "peak_head_t1_angle_deg",
    ]
    display_cols = [c for c in scalar_cols if c in df.columns]

    # Column widths
    col_widths = {}
    for col in display_cols:
        max_val_len = max(
            (len(_fmt_cell(v)) for v in df[col]),
            default=0
        )
        col_widths[col] = max(len(col), max_val_len)

    header = "  ".join(c.ljust(col_widths[c]) for c in display_cols)
    sep = "  ".join("-" * col_widths[c] for c in display_cols)

    print("\n=== Sweep Results Summary ===")
    print(header)
    print(sep)
    for _, row in df.iterrows():
        line = "  ".join(
            _fmt_cell(row[c]).ljust(col_widths[c]) for c in display_cols
        )
        print(line)
    print(sep)
    print(f"Total runs: {len(df)}\n")


def _fmt_cell(val: Any) -> str:
    """Format a single cell value for tabular display."""
    import math
    if val is None:
        return "N/A"
    if isinstance(val, float):
        if math.isnan(val):
            return "NaN"
        return f"{val:.3f}"
    return str(val)


# ---------------------------------------------------------------------------
# Subcommand: validate
# ---------------------------------------------------------------------------

def cmd_validate(args: argparse.Namespace) -> int:
    """
    Parse deck + CSV and print muscle group validation summary.
    """
    _setup_logging()

    (
        DeckParser, NeckMuscleCSVParser, MuscleMapper,
        MUSCLE_GROUP_PATTERNS, ALL_CERVICAL_GROUPS,
        RunConfig, ParametricEditor, load_editor,
        BatchOrchestrator, JobStatus,
        THFileParser, MetricsExtractor, SweepReporter, postprocess_sweep,
    ) = _import_modules()

    deck_path = Path(args.deck)
    csv_path = Path(args.csv)

    # Validate file existence
    if not deck_path.exists():
        logger.error("Deck file not found: %s", deck_path)
        return 1
    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        return 1

    logger.info("Parsing CSV: %s", csv_path)
    try:
        csv_parser = NeckMuscleCSVParser(str(csv_path))
        csv_data = csv_parser.parse()
        csv_rows = len(csv_data) if hasattr(csv_data, "__len__") else "unknown"
        logger.info("CSV parsed: %s rows", csv_rows)
    except Exception as exc:
        logger.error("CSV parsing failed: %s", exc)
        logger.debug(traceback.format_exc())
        return 1

    logger.info("Parsing deck and building muscle mapper via load_editor: %s", deck_path)
    try:
        _editor, parser, mapper = load_editor(str(deck_path))
    except Exception as exc:
        logger.error("Deck parsing / muscle mapper failed: %s", exc)
        logger.debug(traceback.format_exc())
        return 1

    _print_validation_summary(mapper, MUSCLE_GROUP_PATTERNS)
    logger.info("Validation complete.")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: sweep
# ---------------------------------------------------------------------------

def cmd_sweep(args: argparse.Namespace) -> int:
    """
    Full parametric sweep: generate decks -> submit jobs -> extract metrics
    -> write CSV + HTML report.
    """
    output_dir = Path(args.output)
    _setup_logging(output_dir)

    (
        DeckParser, NeckMuscleCSVParser, MuscleMapper,
        MUSCLE_GROUP_PATTERNS, ALL_CERVICAL_GROUPS,
        RunConfig, ParametricEditor, load_editor,
        BatchOrchestrator, JobStatus,
        THFileParser, MetricsExtractor, SweepReporter, postprocess_sweep,
    ) = _import_modules()
    # postprocessor.RunConfig (entity IDs + run metadata for metric extraction).
    from postprocessor import RunConfig as PPRunConfig  # noqa: PLC0415

    deck_path = Path(args.deck)
    csv_path = Path(args.csv)
    config_path = args.config
    dry_run: bool = args.dry_run
    n_parallel: int = args.parallel

    # ------------------------------------------------------------------
    # Step a: Parse deck + CSV
    # ------------------------------------------------------------------
    logger.info("Step 1/8: Parsing deck: %s", deck_path)
    if not deck_path.exists():
        logger.error("Deck file not found: %s", deck_path)
        return 1

    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        return 1

    try:
        csv_parser = NeckMuscleCSVParser(str(csv_path))
        csv_parser.parse()
        logger.info("CSV parsed successfully: %s", csv_path)
    except Exception as exc:
        logger.error("CSV parsing failed: %s", exc)
        logger.debug(traceback.format_exc())
        return 1

    # ------------------------------------------------------------------
    # Step b: Build muscle map + print validation summary
    # ------------------------------------------------------------------
    logger.info("Step 2/8: Building muscle map and validation summary")
    try:
        editor, deck_parser, mapper = load_editor(str(deck_path))
        _print_validation_summary(mapper, MUSCLE_GROUP_PATTERNS)
    except Exception as exc:
        logger.error("MuscleMapper build failed: %s", exc)
        logger.debug(traceback.format_exc())
        return 1

    # ------------------------------------------------------------------
    # Step c: Load sweep_config.json
    # ------------------------------------------------------------------
    logger.info("Step 3/8: Loading sweep config: %s", config_path)
    try:
        configs = _load_sweep_config(config_path, RunConfig)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Config load failed: %s", exc)
        return 1

    if not configs:
        logger.error("No valid run configs found in %s. Aborting.", config_path)
        return 1

    # Capture the raw config entries (keyed by run_id) so postprocessing can read
    # entity-ID fields (head_accel_id, upper/lower_neck_section_id, t1_node_id)
    # that are not part of the deck-generation RunConfig dataclass.
    raw_by_id: dict[str, dict] = {}
    try:
        with open(config_path, "r", encoding="utf-8") as _fh:
            for _entry in json.load(_fh):
                if isinstance(_entry, dict) and "run_id" in _entry:
                    raw_by_id[str(_entry["run_id"])] = _entry
    except (OSError, ValueError):
        pass

    logger.info("%d run configs loaded.", len(configs))

    if dry_run:
        logger.info("[DRY-RUN] Would generate %d decks. Exiting.", len(configs))
        print(f"\n[DRY-RUN] {len(configs)} runs would be submitted:")
        for cfg in configs:
            print(
                f"  {cfg.run_id:20s}  groups={cfg.activated_groups}"
                f"  t_start={cfg.t_start:.3f}s  mag={cfg.magnitude_pct:.0f}%"
                f"  peak_g={cfg.peak_g:.1f}g"
            )
        return 0

    # ------------------------------------------------------------------
    # Step d: Generate modified decks via ParametricEditor
    # ------------------------------------------------------------------
    logger.info("Step 4/8: Generating modified decks into %s", output_dir)
    try:
        deck_paths = editor.generate_batch(
            configs, str(output_dir), base_deck_path=str(deck_path)
        )
    except Exception as exc:
        logger.error("Deck generation failed: %s", exc)
        logger.debug(traceback.format_exc())
        return 1

    logger.info("Generated %d modified decks.", len(deck_paths))

    # Build run_dir list from generated deck paths (parent dirs)
    run_dirs = [str(Path(p).parent) for p in deck_paths]
    # The Starter filename submitted to the solver matches what generate_batch
    # wrote (e.g. <root>_0000.rad); fall back to legacy model.rad otherwise.
    starter_filename = Path(deck_paths[0]).name if deck_paths else "model.rad"

    # ------------------------------------------------------------------
    # Step e: Submit batch via BatchOrchestrator
    # ------------------------------------------------------------------
    logger.info(
        "Step 5/8: Submitting %d jobs (n_parallel=%d)", len(run_dirs), n_parallel
    )
    try:
        orchestrator = BatchOrchestrator(n_parallel=n_parallel)
        status_map = orchestrator.submit_batch(run_dirs, deck_filename=starter_filename)
    except Exception as exc:
        logger.error("Batch submission failed: %s", exc)
        logger.debug(traceback.format_exc())
        return 1

    # ------------------------------------------------------------------
    # Step f: Wait for completion (submit_batch already blocks; this
    #         is a fallback poll in case submit returns before all jobs
    #         reach terminal state)
    # ------------------------------------------------------------------
    logger.info("Step 6/8: Waiting for all jobs to complete")
    try:
        final_statuses = orchestrator.wait_for_all(timeout=7200)
    except Exception as exc:
        logger.warning("wait_for_all raised: %s - using submit status map.", exc)
        final_statuses = status_map

    n_complete = sum(
        1 for s in final_statuses.values()
        if s == JobStatus.COMPLETE
    )
    n_failed = sum(
        1 for s in final_statuses.values()
        if s == JobStatus.FAILED
    )
    logger.info(
        "Batch finished: %d complete, %d failed, %d total",
        n_complete, n_failed, len(final_statuses),
    )

    if n_failed > 0:
        failed_ids = [
            rid for rid, s in final_statuses.items()
            if s == JobStatus.FAILED
        ]
        logger.warning("Failed runs: %s", failed_ids)

    # ------------------------------------------------------------------
    # Step g: Extract metrics from each run's TH output
    # ------------------------------------------------------------------
    logger.info("Step 7/8: Extracting metrics from completed runs")

    # Build run_config lookup by run_id
    config_by_id: dict[str, Any] = {}
    for cfg in configs:
        # run_dir name = run_{run_id}
        config_by_id[f"run_{cfg.run_id}"] = cfg
        # also try without prefix in case run_id already has it
        config_by_id[cfg.run_id] = cfg

    run_list: list[tuple[str, Any]] = []
    skipped_failed: list[str] = []

    for run_dir_str in run_dirs:
        run_dir = Path(run_dir_str)
        run_dir_name = run_dir.name

        # Skip if this run failed
        run_status = final_statuses.get(run_dir_name, JobStatus.FAILED)
        if run_status == JobStatus.FAILED:
            skipped_failed.append(run_dir_name)
            logger.warning(
                "Skipping metric extraction for failed run: %s", run_dir_name
            )
            continue

        # Locate T01 TH file: OpenRadioss names it model_T01 or *.T01 / *_THxx
        th_file = _find_th_file(run_dir)
        if th_file is None:
            logger.warning(
                "No T01 time-history file found in %s - skipping.", run_dir
            )
            skipped_failed.append(run_dir_name)
            continue

        # Look up the original RunConfig
        orig_cfg = config_by_id.get(run_dir_name) or config_by_id.get(
            run_dir_name.replace("run_", "", 1)
        )

        if orig_cfg is not None:
            # Convert ParametricEditor RunConfig -> postprocessor RunConfig.
            # Entity IDs (which accelerometer is the head, which sections are the
            # upper/lower neck) come from the raw config entry; they select the
            # channels to extract from the solved time-history.
            raw = raw_by_id.get(orig_cfg.run_id, {})
            pp_cfg = PPRunConfig(
                run_id=orig_cfg.run_id,
                activated_groups=list(orig_cfg.activated_groups),
                t_start_ms=orig_cfg.t_start * 1000.0,
                magnitude_pct=orig_cfg.magnitude_pct,
                head_part_id=raw.get("head_accel_id") or raw.get("head_part_id"),
                head_node_id=raw.get("head_node_id"),
                t1_node_id=raw.get("t1_node_id"),
                upper_neck_part_id=raw.get("upper_neck_section_id")
                or raw.get("upper_neck_part_id"),
                lower_neck_part_id=raw.get("lower_neck_section_id")
                or raw.get("lower_neck_part_id"),
            )
        else:
            logger.warning(
                "No RunConfig matched for %s - using defaults.", run_dir_name
            )
            pp_cfg = PPRunConfig(run_id=run_dir_name)

        run_list.append((str(th_file), pp_cfg))

    # Add placeholder rows for failed runs so they appear in the report
    all_metrics: list[dict] = []
    for failed_id in skipped_failed:
        all_metrics.append(
            {
                "run_id": failed_id,
                "activated_groups": "FAILED",
                "t_start_ms": float("nan"),
                "magnitude_pct": float("nan"),
                "peak_g": float("nan"),
                "HIC15": float("nan"),
                "peak_head_accel_g": float("nan"),
                "peak_head_t1_angle_deg": float("nan"),
                "upper_neck_Fx_N": float("nan"),
                "upper_neck_Fz_N": float("nan"),
                "upper_neck_My_Nm": float("nan"),
                "lower_neck_Fx_N": float("nan"),
                "lower_neck_Fz_N": float("nan"),
                "lower_neck_My_Nm": float("nan"),
            }
        )

    # Process each successful run individually (so failures don't abort others)
    for th_file_path, pp_cfg in run_list:
        logger.info("Extracting metrics: %s from %s", pp_cfg.run_id, th_file_path)
        try:
            th_parser = THFileParser(th_file_path)
            th_data = th_parser.parse()
            extractor = MetricsExtractor(th_data, pp_cfg)
            m = extractor.extract_all()
            all_metrics.append(m)
        except Exception as exc:
            logger.error(
                "Metric extraction failed for %s: %s", pp_cfg.run_id, exc
            )
            logger.debug(traceback.format_exc())
            all_metrics.append(
                {
                    "run_id": pp_cfg.run_id,
                    "activated_groups": ";".join(pp_cfg.activated_groups),
                    "t_start_ms": pp_cfg.t_start_ms,
                    "magnitude_pct": pp_cfg.magnitude_pct,
                    **{
                        k: float("nan")
                        for k in SweepReporter._SCALAR_COLS
                        if k not in ("run_id", "activated_groups", "t_start_ms", "magnitude_pct")
                    },
                }
            )

    # ------------------------------------------------------------------
    # Step h: Write consolidated CSV and HTML report
    # ------------------------------------------------------------------
    logger.info("Step 8/8: Writing CSV and HTML report")
    report_name = "sweep_report"
    try:
        reporter = SweepReporter(all_metrics)
        csv_out = output_dir / f"{report_name}.csv"
        html_out = output_dir / f"{report_name}.html"
        reporter.to_csv(str(csv_out))
        reporter.to_html_report(str(html_out))
        logger.info("CSV report written: %s", csv_out)
        logger.info("HTML report written: %s", html_out)
    except Exception as exc:
        logger.error("Report generation failed: %s", exc)
        logger.debug(traceback.format_exc())
        return 1

    # ------------------------------------------------------------------
    # Step i: Print final summary table to console
    # ------------------------------------------------------------------
    try:
        df = reporter._build_df()
        _print_final_table(df)
    except Exception as exc:
        logger.warning("Could not print summary table: %s", exc)

    # Final status line
    n_ok = len(run_list)
    n_fail = len(skipped_failed)
    print(
        f"\nSweep complete: {n_ok} succeeded, {n_fail} failed, "
        f"{n_ok + n_fail} total\n"
        f"  CSV  -> {csv_out}\n"
        f"  HTML -> {html_out}\n"
        f"  Log  -> {output_dir / 'sweep.log'}\n"
    )

    return 0 if n_fail == 0 else 2  # 2 = partial failure


# ---------------------------------------------------------------------------
# Subcommand: report
# ---------------------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> int:
    """
    Re-run post-processing on existing completed run directories.
    Discovers all run_* subdirectories under --results-dir, locates their
    T01 files, extracts metrics, and writes a new report to --output.
    """
    output_dir = Path(args.output)
    results_dir = Path(args.results_dir)
    _setup_logging(output_dir)

    (
        DeckParser, NeckMuscleCSVParser, MuscleMapper,
        MUSCLE_GROUP_PATTERNS, ALL_CERVICAL_GROUPS,
        RunConfig, ParametricEditor, load_editor,
        BatchOrchestrator, JobStatus,
        THFileParser, MetricsExtractor, SweepReporter, postprocess_sweep,
    ) = _import_modules()
    # postprocessor.RunConfig (entity IDs + run metadata for metric extraction).
    from postprocessor import RunConfig as PPRunConfig  # noqa: PLC0415

    if not results_dir.exists():
        logger.error("Results directory does not exist: %s", results_dir)
        return 1

    # Discover all run_* subdirectories
    run_dirs = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    )

    if not run_dirs:
        logger.error(
            "No 'run_*' subdirectories found under %s", results_dir
        )
        return 1

    logger.info("Found %d run directories under %s", len(run_dirs), results_dir)

    run_list: list[tuple[str, PPRunConfig]] = []
    skipped: list[str] = []

    for run_dir in run_dirs:
        th_file = _find_th_file(run_dir)
        if th_file is None:
            logger.warning(
                "No T01 file in %s - run may still be in progress or failed.",
                run_dir
            )
            skipped.append(run_dir.name)
            continue

        # Attempt to read RunConfig from run_config.json if present
        cfg = _load_run_config_sidecar(run_dir, run_dir.name, PPRunConfig)
        run_list.append((str(th_file), cfg))

    if not run_list:
        logger.error("No processable runs found. Aborting.")
        return 1

    logger.info("Processing %d runs (%d skipped)", len(run_list), len(skipped))

    # Delegate to postprocess_sweep
    try:
        df = postprocess_sweep(
            run_list=run_list,
            output_dir=str(output_dir),
            report_name="sweep_report",
        )
    except Exception as exc:
        logger.error("postprocess_sweep failed: %s", exc)
        logger.debug(traceback.format_exc())
        return 1

    _print_final_table(df)

    csv_out = output_dir / "sweep_report.csv"
    html_out = output_dir / "sweep_report.html"
    print(
        f"\nReport complete: {len(run_list)} runs processed, "
        f"{len(skipped)} skipped\n"
        f"  CSV  -> {csv_out}\n"
        f"  HTML -> {html_out}\n"
        f"  Log  -> {output_dir / 'sweep.log'}\n"
    )

    return 0


# ---------------------------------------------------------------------------
# Helper: locate the T01 time-history file inside a run directory
# ---------------------------------------------------------------------------

def _find_th_file(run_dir: Path) -> Optional[Path]:
    """
    Search a run directory for OpenRadioss T01 time-history files.

    OpenRadioss names these:
      model_T01   (no extension, standard)
      model0001_T01
      *.T01
      *_TH01

    Returns the first match, or None if not found.
    """
    patterns = [
        "*_T01",
        "*.T01",
        "*_TH01",
        "*T01*",
    ]
    for pattern in patterns:
        matches = list(run_dir.glob(pattern))
        # Filter out directories (T01 outputs are files)
        file_matches = [m for m in matches if m.is_file()]
        if file_matches:
            return file_matches[0]
    return None


# ---------------------------------------------------------------------------
# Helper: load run_config.json sidecar (written alongside the deck by the
# generator if present, otherwise synthesize a minimal config)
# ---------------------------------------------------------------------------

def _load_run_config_sidecar(
    run_dir: Path, fallback_run_id: str, PPRunConfig: Any
) -> Any:
    """
    Look for a run_config.json file inside run_dir and deserialize it.
    Falls back to a minimal PPRunConfig if the file is absent.
    """
    sidecar = run_dir / "run_config.json"
    if sidecar.exists():
        try:
            with open(sidecar, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # Map ParametricEditor RunConfig keys to postprocessor RunConfig keys
            run_id = data.get("run_id", fallback_run_id)
            activated_groups = data.get("activated_groups", [])
            t_start_s = data.get("t_start", 0.0)
            magnitude_pct = data.get("magnitude_pct", 100.0)
            return PPRunConfig(
                run_id=run_id,
                activated_groups=activated_groups,
                t_start_ms=t_start_s * 1000.0,
                magnitude_pct=magnitude_pct,
                head_part_id=data.get("head_accel_id") or data.get("head_part_id"),
                head_node_id=data.get("head_node_id"),
                t1_node_id=data.get("t1_node_id"),
                upper_neck_part_id=data.get("upper_neck_section_id")
                or data.get("upper_neck_part_id"),
                lower_neck_part_id=data.get("lower_neck_section_id")
                or data.get("lower_neck_part_id"),
            )
        except Exception as exc:
            logger.debug(
                "Could not read sidecar config %s: %s - using defaults.", sidecar, exc
            )

    return PPRunConfig(run_id=fallback_run_id)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_sweep.py",
        description=(
            "OpenRadioss parametric sweep CLI.\n\n"
            "Subcommands:\n"
            "  validate  - Check deck and CSV parsing, print muscle group counts\n"
            "  sweep     - Run a full parametric sweep end-to-end\n"
            "  report    - Re-run post-processing on existing completed runs"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    sub = parser.add_subparsers(dest="command", metavar="SUBCOMMAND")
    sub.required = True

    # ------------------------------------------------------------------
    # validate
    # ------------------------------------------------------------------
    p_validate = sub.add_parser(
        "validate",
        help="Validate deck parsing and muscle group mapping.",
    )
    p_validate.add_argument(
        "--deck", required=True, metavar="PATH",
        help="Path to the base OpenRadioss .rad deck file.",
    )
    p_validate.add_argument(
        "--csv", required=True, metavar="PATH",
        help="Path to the neck_muscles.csv file.",
    )

    # ------------------------------------------------------------------
    # sweep
    # ------------------------------------------------------------------
    p_sweep = sub.add_parser(
        "sweep",
        help="Run a full parametric sweep.",
    )
    p_sweep.add_argument(
        "--deck", required=True, metavar="PATH",
        help="Path to the base OpenRadioss .rad deck file.",
    )
    p_sweep.add_argument(
        "--csv", required=True, metavar="PATH",
        help="Path to the neck_muscles.csv file.",
    )
    p_sweep.add_argument(
        "--config", required=True, metavar="PATH",
        default="sweep_config.json",
        help="Path to sweep_config.json (array of RunConfig dicts).",
    )
    p_sweep.add_argument(
        "--output", required=True, metavar="OUTPUT_DIR",
        help="Root directory for run subdirectories, logs, and reports.",
    )
    p_sweep.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate decks and print the run plan, but do not submit jobs.",
    )
    p_sweep.add_argument(
        "--parallel", type=int, default=1, metavar="N",
        help="Number of simultaneous OpenRadioss jobs (default: 1).",
    )

    # ------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------
    p_report = sub.add_parser(
        "report",
        help="Re-run post-processing on existing completed run directories.",
    )
    p_report.add_argument(
        "--results-dir", required=True, metavar="DIR",
        help="Directory containing run_* subdirectories from a previous sweep.",
    )
    p_report.add_argument(
        "--output", required=True, metavar="OUTPUT_DIR",
        help="Directory to write the regenerated CSV and HTML report.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO

    # Dispatch
    if args.command == "validate":
        _setup_logging(level=log_level)
        return cmd_validate(args)
    elif args.command == "sweep":
        return cmd_sweep(args)
    elif args.command == "report":
        return cmd_report(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
