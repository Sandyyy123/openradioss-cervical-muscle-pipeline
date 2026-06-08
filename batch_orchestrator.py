"""
batch_orchestrator.py - OpenRadioss batch job orchestration module.

Manages parallel OpenRadioss simulation runs, monitors job status,
captures output, and reports progress via tqdm.
"""

from __future__ import annotations

import enum
import glob
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # graceful degradation if tqdm not installed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and data structures
# ---------------------------------------------------------------------------

class JobStatus(enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass
class JobRecord:
    """Internal record for a single OpenRadioss job."""
    run_id: str
    run_dir: str
    deck_filename: str
    status: JobStatus = JobStatus.PENDING
    pid: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    returncode: Optional[int] = None
    log_path: Optional[str] = None
    error_message: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @property
    def elapsed(self) -> float:
        """Elapsed wall-clock seconds, or 0 if not started."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class BatchOrchestrator:
    """
    Manages parallel OpenRadioss simulation jobs.

    Parameters
    ----------
    openradioss_executable : str
        Path to the OpenRadioss engine binary, or just ``"OpenRadioss"`` if
        it is available on ``PATH``.
    n_parallel : int
        Maximum number of jobs to run simultaneously.
    simulate : bool
        Dry-run mode - print commands without executing.  Useful for CI
        environments where OpenRadioss is not installed.
    nproc : int
        Number of MPI processes / OpenMP threads to pass via ``-np``.
    """

    def __init__(
        self,
        openradioss_executable: str = "OpenRadioss",
        n_parallel: int = 1,
        simulate: bool = False,
        nproc: int = 1,
    ) -> None:
        self.executable = openradioss_executable
        self.n_parallel = max(1, n_parallel)
        self.simulate = simulate
        self.nproc = max(1, nproc)

        self._jobs: dict[str, JobRecord] = {}
        self._jobs_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit_batch(
        self,
        run_dirs: list[str],
        deck_filename: str = "model.rad",
    ) -> dict[str, JobStatus]:
        """
        Submit all run directories as a batch.

        Runs up to ``n_parallel`` jobs concurrently using a thread pool.
        Each job streams stdout/stderr to ``run.log`` inside its run_dir.

        Parameters
        ----------
        run_dirs :
            List of directories, each containing a valid OpenRadioss deck.
        deck_filename :
            Name of the input deck file inside each run_dir.

        Returns
        -------
        dict[str, JobStatus]
            Mapping of run_id -> final JobStatus for every submitted job.
        """
        if not run_dirs:
            logger.warning("submit_batch called with empty run_dirs list.")
            return {}

        # Register all jobs as PENDING
        records: list[JobRecord] = []
        for run_dir in run_dirs:
            run_id = Path(run_dir).resolve().name
            # Handle duplicate names by appending index
            if run_id in self._jobs:
                run_id = f"{run_id}_{len(self._jobs)}"
            record = JobRecord(
                run_id=run_id,
                run_dir=str(Path(run_dir).resolve()),
                deck_filename=deck_filename,
            )
            with self._jobs_lock:
                self._jobs[run_id] = record
            records.append(record)

        logger.info(
            "Submitting %d job(s), n_parallel=%d, simulate=%s",
            len(records),
            self.n_parallel,
            self.simulate,
        )

        pbar = None
        if tqdm is not None:
            pbar = tqdm(total=len(records), desc="OpenRadioss batch", unit="job")

        with ThreadPoolExecutor(max_workers=self.n_parallel) as pool:
            future_to_record = {
                pool.submit(self._run_job, rec): rec for rec in records
            }
            for future in as_completed(future_to_record):
                rec = future_to_record[future]
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001
                    with rec.lock:
                        rec.status = JobStatus.FAILED
                        rec.error_message = str(exc)
                    logger.exception("Unexpected error in job %s", rec.run_id)
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(
                        last=rec.run_id,
                        status=rec.status.value,
                        elapsed=f"{rec.elapsed:.0f}s",
                    )

        if pbar is not None:
            pbar.close()

        return {rec.run_id: rec.status for rec in records}

    def run_single(
        self,
        run_dir: str,
        deck_filename: str = "model.rad",
        timeout: int = 3600,
    ) -> JobStatus:
        """
        Run a single OpenRadioss job synchronously and return its status.

        Parameters
        ----------
        run_dir :
            Directory containing the input deck.
        deck_filename :
            Name of the input deck file.
        timeout :
            Hard wall-clock timeout in seconds.  The subprocess is killed if
            it exceeds this limit and the job is marked FAILED.

        Returns
        -------
        JobStatus
        """
        run_id = Path(run_dir).resolve().name
        if run_id in self._jobs:
            run_id = f"{run_id}_{len(self._jobs)}"

        record = JobRecord(
            run_id=run_id,
            run_dir=str(Path(run_dir).resolve()),
            deck_filename=deck_filename,
        )
        with self._jobs_lock:
            self._jobs[run_id] = record

        self._run_job(record, timeout=timeout)
        return record.status

    def wait_for_all(self, timeout: int = 7200) -> dict[str, JobStatus]:
        """
        Block until all registered jobs reach a terminal state or *timeout*
        seconds elapse.

        Polls every 30 seconds and prints a progress summary to stdout.

        Parameters
        ----------
        timeout :
            Maximum total wait time in seconds.

        Returns
        -------
        dict[str, JobStatus]
            Current status snapshot for every registered job.
        """
        deadline = time.time() + timeout
        poll_interval = 30

        while time.time() < deadline:
            with self._jobs_lock:
                snapshot = {jid: rec.status for jid, rec in self._jobs.items()}

            running = sum(1 for s in snapshot.values() if s == JobStatus.RUNNING)
            pending = sum(1 for s in snapshot.values() if s == JobStatus.PENDING)
            complete = sum(1 for s in snapshot.values() if s == JobStatus.COMPLETE)
            failed = sum(1 for s in snapshot.values() if s == JobStatus.FAILED)

            logger.info(
                "Progress - pending:%d running:%d complete:%d failed:%d",
                pending, running, complete, failed,
            )
            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"pending={pending} running={running} "
                f"complete={complete} failed={failed}"
            )

            if pending == 0 and running == 0:
                break

            time.sleep(poll_interval)
        else:
            logger.warning("wait_for_all timed out after %ds", timeout)

        with self._jobs_lock:
            return {jid: rec.status for jid, rec in self._jobs.items()}

    def get_output_files(self, run_dir: str) -> dict:
        """
        Discover OpenRadioss output files inside *run_dir*.

        OpenRadioss naming conventions
        --------------------------------
        - ``*T01``        : time-history (ASCII) file
        - ``*A001``, ``*A002``, ... : animation files
        - ``*_0001.rad``  : restart file
        - ``run.log``     : stdout/stderr captured by this orchestrator

        Parameters
        ----------
        run_dir :
            Directory to inspect.

        Returns
        -------
        dict with keys:
            ``th_file``    - str path to the time-history file, or None
            ``anim_files`` - sorted list of animation file paths
            ``restart_files`` - sorted list of restart file paths
            ``log``        - str path to run.log, or None
        """
        base = Path(run_dir).resolve()

        # Time-history: ends with T01 (or T02, etc. but T01 is primary)
        th_candidates = sorted(base.glob("*T01"))
        th_file = str(th_candidates[0]) if th_candidates else None

        # Animation files: *A001, *A002 ...
        anim_files = sorted(base.glob("*A[0-9][0-9][0-9]"))

        # Restart files: *_0001.rad
        restart_files = sorted(base.glob("*_[0-9]*.rad"))

        log_path = base / "run.log"
        log_file = str(log_path) if log_path.exists() else None

        return {
            "th_file": th_file,
            "anim_files": [str(p) for p in anim_files],
            "restart_files": [str(p) for p in restart_files],
            "log": log_file,
        }

    # ------------------------------------------------------------------
    # Properties / status helpers
    # ------------------------------------------------------------------

    @property
    def all_statuses(self) -> dict[str, JobStatus]:
        """Return a snapshot of every registered job's status."""
        with self._jobs_lock:
            return {jid: rec.status for jid, rec in self._jobs.items()}

    def summary(self) -> str:
        """Return a human-readable one-line summary of batch progress."""
        statuses = list(self.all_statuses.values())
        counts = {s: statuses.count(s) for s in JobStatus}
        parts = [f"{s.value.lower()}={counts[s]}" for s in JobStatus if counts[s] > 0]
        return "BatchOrchestrator: " + "  ".join(parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_command(self, record: JobRecord) -> list[str]:
        """Assemble the OpenRadioss CLI command for the given job record."""
        return [
            self.executable,
            "-i", record.deck_filename,
            "-np", str(self.nproc),
        ]

    def _run_job(self, record: JobRecord, timeout: int = 3600) -> None:
        """
        Execute a single job.  Called from a worker thread.

        Updates ``record.status`` throughout.  Writes combined stdout+stderr
        to ``<run_dir>/run.log``.
        """
        cmd = self._build_command(record)
        run_dir = record.run_dir
        log_path = os.path.join(run_dir, "run.log")
        record.log_path = log_path

        if self.simulate:
            self._simulate_job(record, cmd)
            return

        # Validate run directory
        if not os.path.isdir(run_dir):
            with record.lock:
                record.status = JobStatus.FAILED
                record.error_message = f"run_dir does not exist: {run_dir}"
            logger.error("Job %s failed: %s", record.run_id, record.error_message)
            return

        deck_path = os.path.join(run_dir, record.deck_filename)
        if not os.path.isfile(deck_path):
            with record.lock:
                record.status = JobStatus.FAILED
                record.error_message = f"deck file not found: {deck_path}"
            logger.error("Job %s failed: %s", record.run_id, record.error_message)
            return

        with record.lock:
            record.status = JobStatus.RUNNING
            record.start_time = time.time()

        logger.info(
            "Starting job %s: %s (cwd=%s)", record.run_id, " ".join(cmd), run_dir
        )

        proc = None
        try:
            with open(log_path, "w", encoding="utf-8", errors="replace") as log_fh:
                log_fh.write(f"# OpenRadioss job: {record.run_id}\n")
                log_fh.write(f"# Command: {' '.join(cmd)}\n")
                log_fh.write(f"# Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                log_fh.flush()

                proc = subprocess.Popen(
                    cmd,
                    cwd=run_dir,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                with record.lock:
                    record.pid = proc.pid

                try:
                    returncode = proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    with record.lock:
                        record.status = JobStatus.FAILED
                        record.returncode = -1
                        record.end_time = time.time()
                        record.error_message = (
                            f"Job exceeded timeout of {timeout}s and was killed."
                        )
                    log_fh.write(
                        f"\n# TIMEOUT after {timeout}s - process killed.\n"
                    )
                    logger.warning(
                        "Job %s timed out after %ds", record.run_id, timeout
                    )
                    return

            with record.lock:
                record.returncode = returncode
                record.end_time = time.time()

            if returncode == 0:
                # Extra check: did OpenRadioss actually produce output?
                outputs = self.get_output_files(run_dir)
                has_output = bool(outputs["th_file"] or outputs["anim_files"])
                if has_output:
                    with record.lock:
                        record.status = JobStatus.COMPLETE
                    logger.info(
                        "Job %s COMPLETE in %.1fs", record.run_id, record.elapsed
                    )
                else:
                    with record.lock:
                        record.status = JobStatus.FAILED
                        record.error_message = (
                            "Process exited 0 but no output files (*T01 / *A001) found."
                        )
                    logger.warning(
                        "Job %s returned 0 but no output files found.", record.run_id
                    )
            else:
                with record.lock:
                    record.status = JobStatus.FAILED
                    record.error_message = (
                        f"Process exited with non-zero returncode: {returncode}"
                    )
                logger.error(
                    "Job %s FAILED (rc=%d) in %.1fs",
                    record.run_id, returncode, record.elapsed,
                )

        except FileNotFoundError:
            with record.lock:
                record.status = JobStatus.FAILED
                record.end_time = time.time()
                record.error_message = (
                    f"Executable not found: '{self.executable}'. "
                    "Ensure OpenRadioss is installed and on PATH, "
                    "or pass the full path via openradioss_executable."
                )
            logger.error("Job %s failed: %s", record.run_id, record.error_message)

        except Exception as exc:  # noqa: BLE001
            with record.lock:
                record.status = JobStatus.FAILED
                record.end_time = time.time()
                record.error_message = str(exc)
            logger.exception("Job %s raised unexpected exception", record.run_id)
            if proc is not None:
                try:
                    proc.kill()
                except OSError:
                    pass

    def _simulate_job(self, record: JobRecord, cmd: list[str]) -> None:
        """Dry-run: print the command and mark the job COMPLETE immediately."""
        with record.lock:
            record.status = JobStatus.RUNNING
            record.start_time = time.time()

        print(
            f"[DRY-RUN] Job '{record.run_id}' "
            f"would execute in '{record.run_dir}':\n"
            f"  {' '.join(cmd)}"
        )
        logger.info(
            "DRY-RUN: job %s command: %s (cwd=%s)",
            record.run_id, " ".join(cmd), record.run_dir,
        )

        # Write a placeholder log so callers can inspect it
        log_path = os.path.join(record.run_dir, "run.log")
        try:
            os.makedirs(record.run_dir, exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write(f"# DRY-RUN mode - no process was executed.\n")
                fh.write(f"# Command: {' '.join(cmd)}\n")
        except OSError:
            pass  # run_dir may not exist in dry-run; that is acceptable

        with record.lock:
            record.status = JobStatus.COMPLETE
            record.end_time = time.time()
            record.log_path = log_path
            record.returncode = 0


# ---------------------------------------------------------------------------
# Convenience CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    """Minimal CLI: python -m batch_orchestrator <run_dir1> [run_dir2 ...]"""
    import argparse

    parser = argparse.ArgumentParser(
        description="OpenRadioss batch orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_dirs", nargs="+", help="Run directories to process")
    parser.add_argument("--deck", default="model.rad", help="Input deck filename")
    parser.add_argument("--exe", default="OpenRadioss", help="Path to OpenRadioss binary")
    parser.add_argument("--n-parallel", type=int, default=1, help="Parallel jobs")
    parser.add_argument("--nproc", type=int, default=1, help="MPI ranks / OpenMP threads")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-job timeout (s)")
    parser.add_argument("--simulate", action="store_true", help="Dry-run mode")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    orch = BatchOrchestrator(
        openradioss_executable=args.exe,
        n_parallel=args.n_parallel,
        simulate=args.simulate,
        nproc=args.nproc,
    )

    results = orch.submit_batch(args.run_dirs, deck_filename=args.deck)

    print("\n--- Batch Results ---")
    for run_id, status in results.items():
        print(f"  {run_id}: {status.value}")

    failed = [rid for rid, s in results.items() if s == JobStatus.FAILED]
    if failed:
        print(f"\n{len(failed)} job(s) FAILED: {', '.join(failed)}")
        raise SystemExit(1)


if __name__ == "__main__":
    _cli()
