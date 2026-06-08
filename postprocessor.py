"""
postprocessor.py
OpenRadioss T01 time-history postprocessor for rear-impact biomechanics studies.
Extracts HIC15, peak head acceleration, head-T1 angle, and neck loads.
Designed for Yoganandan 2000 sled setup with VIVA+ HBM.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import os
import re
import shutil
import subprocess
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

G_MS2 = 9.81  # m/s^2 per g


# ---------------------------------------------------------------------------
# Configuration dataclass used by MetricsExtractor
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    run_id: str
    activated_groups: list[str] = field(default_factory=list)
    t_start_ms: float = 0.0          # pulse onset time, milliseconds
    magnitude_pct: float = 100.0     # pulse magnitude as % of reference
    head_node_id: Optional[str] = None
    t1_node_id: Optional[str] = None
    head_part_id: Optional[str] = None
    upper_neck_part_id: Optional[str] = None
    lower_neck_part_id: Optional[str] = None


# ---------------------------------------------------------------------------
# THFileParser
# ---------------------------------------------------------------------------

class THFileParser:
    """
    Parses an OpenRadioss ASCII T01 time-history file.

    OpenRadioss T01 format (simplified):
        - Comment / header lines starting with '#'
        - Entity-block header:  ENTITY_TYPE  ENTITY_ID  ENTITY_TITLE
          e.g.  NODE  1234  HEAD_CG
        - Signal-name line:  TIME  AX  AY  AZ  ...
        - Data rows (floating-point columns matching signal names)
        - Blank line or next entity header terminates the block

    Returns a dict keyed by entity label ->
        pd.DataFrame with column "time" plus one column per signal.
    """

    # Regex: entity header line (not a pure data line)
    _ENTITY_RE = re.compile(
        r"^\s*(NODE|PART|SECTION|GROUP|SPRING|BEAM|SHELL|SOLID|RBODY|"
        r"SENSOR|ACCEL|FORCE|MOMENT|RIGID)\s+(\S+)\s*(.*)?$",
        re.IGNORECASE,
    )
    # Numeric data line: starts with (optional whitespace then) a float/int
    _DATA_RE = re.compile(r"^\s*[-+]?\d")

    def __init__(self, th_file_path: str) -> None:
        self.path = Path(th_file_path)
        if not self.path.exists():
            raise FileNotFoundError(f"T01 file not found: {self.path}")
        self._raw: dict[str, pd.DataFrame] = {}
        self._parsed = False
        # OpenRadioss writes a *binary* T01. If we were handed one, convert it to
        # ASCII CSV with the bundled th_to_csv utility and parse that instead.
        self._csv_path = self._ensure_csv(self.path)

    # ------------------------------------------------------------------
    # Binary T01 -> CSV conversion (th_to_csv)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_binary(path: Path) -> bool:
        try:
            with path.open("rb") as fh:
                return b"\x00" in fh.read(4096)
        except OSError:
            return False

    def _ensure_csv(self, path: Path) -> Path:
        """Return a path to an ASCII CSV time-history, converting if needed."""
        if path.suffix.lower() == ".csv":
            return path
        if not self._is_binary(path):
            return path  # already ASCII (legacy text T01)
        csv_path = path.with_name(path.name + ".csv")
        if csv_path.exists():
            return csv_path
        exe = (
            shutil.which("th_to_csv_linux64_gf")
            or shutil.which("th_to_csv")
            or shutil.which("th_to_csv_linux64_gf_sp")
        )
        if not exe:
            logger.warning(
                "Binary T01 %s found but no th_to_csv converter on PATH; "
                "metric extraction will be skipped.", path.name
            )
            return path
        try:
            subprocess.run(
                [exe, path.name], cwd=str(path.parent),
                capture_output=True, text=True, timeout=300, check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("th_to_csv failed on %s: %s", path.name, exc)
            return path
        return csv_path if csv_path.exists() else path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self) -> dict[str, pd.DataFrame]:
        """
        Parse the T01 file and return entity -> DataFrame mapping.
        Re-uses cached result on repeated calls.
        """
        if self._parsed:
            return self._raw
        if self._csv_path.suffix.lower() == ".csv":
            self._raw = self._parse_th_to_csv(self._csv_path)
        else:
            self._raw = self._parse_file()
        self._parsed = True
        logger.info("Parsed %d entities from %s", len(self._raw), self._csv_path.name)
        return self._raw

    # ------------------------------------------------------------------
    # th_to_csv wide-format parser
    # ------------------------------------------------------------------

    # Map a th_to_csv column descriptor to a canonical entity type token.
    _ENTITY_KIND = [
        ("ACCEL", "ACCEL"),
        ("RIGID", "RBODY"),
        ("NODES", "NODE"),
        ("NODE", "NODE"),
        ("SECTIO", "SECTION"),
        ("SECTION", "SECTION"),
        ("SHELL", "SHELL"),
        ("BRICK", "SOLID"),
        ("SOLID", "SOLID"),
        ("SPRING", "SPRING"),
        ("BEAM", "BEAM"),
    ]

    # Positional component names per OpenRadioss time-history entity type.
    # Accelerometers always store AX,AY,AZ; section forces store the standard
    # force/moment sextet. These orders are solver conventions, not deck-specific.
    _COMPONENT_ORDER = {
        "ACCEL": ["AX", "AY", "AZ"],
        "SECTION": ["FX", "FY", "FZ", "MX", "MY", "MZ"],
    }

    @classmethod
    def _classify_column(cls, header: str):
        """Return (entity_type, entity_id) for a th_to_csv column header, or (None, None)."""
        up = header.upper()
        if up.startswith("NULL"):
            return None, None
        kind = None
        for needle, token in cls._ENTITY_KIND:
            if needle in up:
                kind = token
                break
        if kind is None:
            return None, None
        m = re.search(r"(\d+)", header)
        if not m:
            return None, None
        return kind, m.group(1)

    def _component_names(self, kind: str, n: int) -> list:
        base = self._COMPONENT_ORDER.get(kind, [])
        names = list(base[:n])
        # Pad any extra columns with positional names so nothing is dropped.
        for i in range(len(names), n):
            names.append(f"c{i}")
        return names

    def _parse_th_to_csv(self, csv_path: Path) -> dict:
        """Parse a th_to_csv wide CSV into {entity_label: DataFrame(time, components...)}."""
        df = pd.read_csv(csv_path)
        if df.shape[1] < 2:
            return {}
        cols = list(df.columns)
        time_col = cols[0]
        time = pd.to_numeric(df[time_col], errors="coerce")

        groups: "OrderedDict[tuple, list]" = OrderedDict()
        for c in cols[1:]:
            kind, eid = self._classify_column(str(c))
            if kind is None:
                continue
            groups.setdefault((kind, eid), []).append(c)

        result: dict = {}
        for (kind, eid), ccols in groups.items():
            names = self._component_names(kind, len(ccols))
            sub = pd.DataFrame({"time": time})
            for name, c in zip(names, ccols):
                sub[name] = pd.to_numeric(df[c], errors="coerce")
            result[f"{kind}_{eid}"] = sub
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_file(self) -> dict[str, pd.DataFrame]:
        result: dict[str, pd.DataFrame] = {}

        current_label: Optional[str] = None
        current_signals: list[str] = []
        current_rows: list[list[float]] = []
        in_signals_line = False

        def _flush():
            nonlocal current_label, current_signals, current_rows, in_signals_line
            if current_label and current_signals and current_rows:
                df = pd.DataFrame(current_rows, columns=current_signals)
                # Ensure 'time' column present
                if "time" not in df.columns and df.columns[0].lower() in ("time", "t"):
                    df = df.rename(columns={df.columns[0]: "time"})
                result[current_label] = df.apply(pd.to_numeric, errors="coerce")
            current_label = None
            current_signals = []
            current_rows = []
            in_signals_line = False

        with self.path.open("r", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.rstrip("\n")

                # Skip pure comment / empty lines while not inside a block
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                # Try entity header
                m = self._ENTITY_RE.match(line)
                if m:
                    _flush()
                    entity_type = m.group(1).upper()
                    entity_id = m.group(2).strip()
                    entity_title = m.group(3).strip() if m.group(3) else ""
                    current_label = (
                        f"{entity_type}_{entity_id}"
                        if not entity_title
                        else f"{entity_type}_{entity_id}_{entity_title}"
                    )
                    in_signals_line = True  # next non-data line is signal names
                    continue

                # If we are waiting for the signal-name line
                if in_signals_line and not self._DATA_RE.match(line):
                    current_signals = stripped.split()
                    # Normalise first column to 'time'
                    if current_signals and current_signals[0].lower() in ("time", "t", "times"):
                        current_signals[0] = "time"
                    in_signals_line = False
                    continue

                # Data row
                if self._DATA_RE.match(line) and current_label:
                    try:
                        vals = [float(v) for v in stripped.split()]
                        if vals:
                            current_rows.append(vals)
                    except ValueError:
                        pass
                    continue

        _flush()
        return result

    # ------------------------------------------------------------------
    # Convenience: find entity by partial name / id
    # ------------------------------------------------------------------

    def find(self, keyword: str) -> Optional[pd.DataFrame]:
        """Return the first DataFrame whose key contains keyword (case-insensitive)."""
        kw = keyword.lower()
        for k, v in self._raw.items():
            if kw in k.lower():
                return v
        return None

    def keys(self) -> list[str]:
        return list(self._raw.keys())


# ---------------------------------------------------------------------------
# MetricsExtractor
# ---------------------------------------------------------------------------

class MetricsExtractor:
    """
    Extracts biomechanical metrics from parsed T01 time-history data.

    Parameters
    ----------
    th_data : dict[str, pd.DataFrame]
        Output of THFileParser.parse().
    run_config : RunConfig
        Identifiers for head, neck, and T1 entities plus run metadata.
    """

    def __init__(self, th_data: dict[str, pd.DataFrame], run_config: RunConfig) -> None:
        self.th = th_data
        self.cfg = run_config

    # ------------------------------------------------------------------
    # HIC15
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_cfc1000(signal: np.ndarray, dt: float) -> np.ndarray:
        """Apply CFC 1000 Butterworth low-pass filter per SAE J211.

        CFC 1000 specifies a cutoff frequency of 1650 Hz (SAE J211).
        A 2nd-order Butterworth filter is applied zero-phase (filtfilt).
        If the sampling rate is too low to support the cutoff, the signal
        is returned unfiltered.

        Parameters
        ----------
        signal : np.ndarray
            Input acceleration array (g-units).
        dt : float
            Sample interval in seconds.

        Returns
        -------
        np.ndarray
            Filtered signal (or original if fs < 2 * fc).
        """
        from scipy.signal import butter, filtfilt
        fs = 1.0 / dt
        fc = 1650.0  # CFC 1000 cutoff per SAE J211
        if fs < 2 * fc:
            return signal  # sampling rate too low to filter
        b, a = butter(2, fc / (fs / 2), btype="low")
        return filtfilt(b, a, signal)

    @staticmethod
    def extract_hic15(
        head_accel_resultant: pd.Series,
        time: pd.Series,
        window_s: float = 0.015,
    ) -> float:
        """
        Compute HIC15.

        HIC15 = max_{t1,t2: (t2-t1) <= 0.015 s} of
                (t2-t1) * [ (1/(t2-t1)) * integral_t1^t2 a(t) dt ]^2.5

        Acceleration must be supplied in g-units.
        Uses a vectorised sliding-window approach with trapezoidal integration.

        Signal is pre-filtered with CFC 1000 (SAE J211, Butterworth 2nd order,
        fc=1650 Hz) before HIC integration.

        Returns
        -------
        float
            HIC15 value (dimensionless).
        """
        a = np.asarray(head_accel_resultant, dtype=float)
        t = np.asarray(time, dtype=float)

        if len(t) < 2:
            return 0.0

        dt_val = float(t[1] - t[0])
        a = MetricsExtractor._apply_cfc1000(a, dt_val)

        # Compute cumulative integral via trapezoid rule
        cumint = np.zeros(len(t))
        cumint[1:] = np.cumsum(0.5 * (a[:-1] + a[1:]) * np.diff(t))

        hic_max = 0.0
        j_start = 0

        for i in range(len(t)):
            # Advance j_start so t[i] - t[j_start] <= window_s
            while t[i] - t[j_start] > window_s + 1e-12:
                j_start += 1

            # All j in [j_start, i] satisfy the window constraint
            # For each such j we want: (t[i]-t[j]) * mean_a^2.5
            for j in range(j_start, i):
                dt = t[i] - t[j]
                if dt <= 0:
                    continue
                mean_a = (cumint[i] - cumint[j]) / dt
                if mean_a <= 0:
                    continue
                hic = dt * (mean_a ** 2.5)
                if hic > hic_max:
                    hic_max = hic

        return float(hic_max)

    @staticmethod
    def extract_hic15_fast(
        head_accel_resultant: pd.Series,
        time: pd.Series,
        window_s: float = 0.015,
    ) -> float:
        """
        Vectorised HIC15 using numpy broadcasting.
        Suitable for large datasets; trades memory for speed.
        Falls back to the safe loop version for very large arrays (>5000 pts).

        Signal is pre-filtered with CFC 1000 (SAE J211, Butterworth 2nd order,
        fc=1650 Hz) before HIC integration.
        """
        a = np.asarray(head_accel_resultant, dtype=float)
        t = np.asarray(time, dtype=float)
        n = len(t)
        if n < 2:
            return 0.0

        dt_val = float(t[1] - t[0]) if n > 1 else 1e-4
        a = MetricsExtractor._apply_cfc1000(a, dt_val)

        if n > 5000:
            # Downsample to avoid excessive memory use
            factor = math.ceil(n / 5000)
            a = a[::factor]
            t = t[::factor]
            n = len(t)

        cumint = np.zeros(n)
        cumint[1:] = np.cumsum(0.5 * (a[:-1] + a[1:]) * np.diff(t))

        # Broadcasting: i index along rows, j along columns
        ti = t[:, None]  # (n, 1)
        tj = t[None, :]  # (1, n)
        dt_mat = ti - tj  # (n, n); positive when i > j

        ci = cumint[:, None]
        cj = cumint[None, :]
        int_mat = ci - cj

        # Mask: 0 < dt <= window_s
        mask = (dt_mat > 0) & (dt_mat <= window_s + 1e-12)
        dt_safe = np.where(mask, dt_mat, 1.0)  # avoid div-by-zero
        mean_a = np.where(mask, int_mat / dt_safe, 0.0)
        mean_a = np.maximum(mean_a, 0.0)

        hic_mat = np.where(mask, dt_mat * (mean_a ** 2.5), 0.0)
        return float(hic_mat.max())

    # ------------------------------------------------------------------
    # Peak head acceleration
    # ------------------------------------------------------------------

    @staticmethod
    def extract_peak_head_accel(head_accel_resultant: pd.Series) -> float:
        """
        Return peak resultant acceleration in g.

        Parameters
        ----------
        head_accel_resultant : pd.Series
            Resultant acceleration time-series already in g-units.
        """
        arr = np.asarray(head_accel_resultant, dtype=float)
        return float(np.nanmax(np.abs(arr)))

    # ------------------------------------------------------------------
    # Head - T1 angle
    # ------------------------------------------------------------------

    @staticmethod
    def extract_head_t1_angle(
        head_node_data: pd.DataFrame,
        t1_node_data: pd.DataFrame,
        rx_col: str = "RX",
        ry_col: str = "RY",
        rz_col: str = "RZ",
    ) -> pd.Series:
        """
        Compute relative rotation angle between the head node and T1 node.

        Expects both DataFrames to contain rotation columns RX, RY, RZ (in degrees
        or radians - detected automatically by magnitude).  Returns relative angle
        magnitude (degrees) interpolated to the head_node_data time grid.

        Returns
        -------
        pd.Series
            Time-series of relative head-T1 angle in degrees, indexed by time.
        """
        def _get_cols(df: pd.DataFrame, rx: str, ry: str, rz: str):
            cols = {c.upper(): c for c in df.columns}
            rx_k = cols.get(rx.upper()) or cols.get("RX") or cols.get("ROTX") or cols.get("ROT_X")
            ry_k = cols.get(ry.upper()) or cols.get("RY") or cols.get("ROTY") or cols.get("ROT_Y")
            rz_k = cols.get(rz.upper()) or cols.get("RZ") or cols.get("ROTZ") or cols.get("ROT_Z")
            return rx_k, ry_k, rz_k

        h_rx, h_ry, h_rz = _get_cols(head_node_data, rx_col, ry_col, rz_col)
        t1_rx, t1_ry, t1_rz = _get_cols(t1_node_data, rx_col, ry_col, rz_col)

        if not all([h_rx, h_ry, h_rz, t1_rx, t1_ry, t1_rz]):
            logger.warning("Rotation columns not found; returning zero series.")
            return pd.Series(np.zeros(len(head_node_data)), index=head_node_data["time"])

        t_head = head_node_data["time"].values
        t_t1 = t1_node_data["time"].values

        def _to_rad(df, col):
            arr = df[col].values.astype(float)
            # Heuristic: if values exceed 2*pi, assume degrees
            if np.nanmax(np.abs(arr)) > 2 * math.pi:
                arr = np.deg2rad(arr)
            return arr

        h_rx_r = _to_rad(head_node_data, h_rx)
        h_ry_r = _to_rad(head_node_data, h_ry)
        h_rz_r = _to_rad(head_node_data, h_rz)

        # Interpolate T1 rotations onto head time grid
        t1_rx_i = np.interp(t_head, t_t1, _to_rad(t1_node_data, t1_rx))
        t1_ry_i = np.interp(t_head, t_t1, _to_rad(t1_node_data, t1_ry))
        t1_rz_i = np.interp(t_head, t_t1, _to_rad(t1_node_data, t1_rz))

        # Relative rotation components
        dRx = h_rx_r - t1_rx_i
        dRy = h_ry_r - t1_ry_i
        dRz = h_rz_r - t1_rz_i

        angle_rad = np.sqrt(dRx**2 + dRy**2 + dRz**2)
        angle_deg = np.rad2deg(angle_rad)

        return pd.Series(angle_deg, index=t_head, name="head_t1_angle_deg")

    # ------------------------------------------------------------------
    # Neck loads
    # ------------------------------------------------------------------

    @staticmethod
    def extract_neck_loads(
        upper_neck_th: pd.DataFrame,
        lower_neck_th: pd.DataFrame,
        fx_col: str = "FX",
        fz_col: str = "FZ",
        my_col: str = "MY",
    ) -> dict:
        """
        Extract peak neck loads from upper (OC joint) and lower (C7/T1) sections.

        Columns are matched case-insensitively.  If a column is absent, the
        corresponding peak is returned as NaN with a warning.

        Returns
        -------
        dict with keys:
            upper_neck_Fx_N, upper_neck_Fz_N, upper_neck_My_Nm,
            lower_neck_Fx_N, lower_neck_Fz_N, lower_neck_My_Nm
        """

        def _col(df: pd.DataFrame, name: str) -> Optional[str]:
            mapping = {c.upper(): c for c in df.columns}
            return mapping.get(name.upper())

        def _peak(df: pd.DataFrame, col_name: str, label: str) -> float:
            col = _col(df, col_name)
            if col is None:
                logger.warning("Column %s not found in %s data", col_name, label)
                return float("nan")
            arr = np.abs(df[col].values.astype(float))
            return float(np.nanmax(arr))

        return {
            "upper_neck_Fx_N": _peak(upper_neck_th, fx_col, "upper_neck"),
            "upper_neck_Fz_N": _peak(upper_neck_th, fz_col, "upper_neck"),
            "upper_neck_My_Nm": _peak(upper_neck_th, my_col, "upper_neck"),
            "lower_neck_Fx_N": _peak(lower_neck_th, fx_col, "lower_neck"),
            "lower_neck_Fz_N": _peak(lower_neck_th, fz_col, "lower_neck"),
            "lower_neck_My_Nm": _peak(lower_neck_th, my_col, "lower_neck"),
        }

    # ------------------------------------------------------------------
    # Resultant acceleration helper
    # ------------------------------------------------------------------

    @staticmethod
    def compute_resultant_g(df: pd.DataFrame,
                            ax_col: str = "AX",
                            ay_col: str = "AY",
                            az_col: str = "AZ") -> pd.Series:
        """
        Compute resultant acceleration magnitude from component columns.
        If components are in m/s^2, converts to g automatically (heuristic: >50 m/s^2).
        """
        col_map = {c.upper(): c for c in df.columns}

        def _get(name):
            c = col_map.get(name.upper())
            if c:
                return df[c].values.astype(float)
            return np.zeros(len(df))

        ax = _get(ax_col)
        ay = _get(ay_col)
        az = _get(az_col)

        res = np.sqrt(ax**2 + ay**2 + az**2)

        # Auto-detect units: if peak > 50, assume m/s^2 and convert
        if np.nanmax(res) > 50.0:
            res = res / G_MS2

        return pd.Series(res, name="resultant_g")

    # ------------------------------------------------------------------
    # Master extractor
    # ------------------------------------------------------------------

    def extract_all(
        self,
        head_part_id: Optional[str] = None,
        upper_neck_part_id: Optional[str] = None,
        lower_neck_part_id: Optional[str] = None,
    ) -> dict:
        """
        Run all extractions and return a consolidated metrics dict.

        Entity IDs can be passed directly or taken from self.cfg.
        Uses partial-name matching against parsed entity keys if exact key
        is not found.

        Returns
        -------
        dict
            Keys: run_id, config fields, and all biomechanical metrics.
        """
        # Resolve IDs
        head_id = head_part_id or self.cfg.head_part_id or self.cfg.head_node_id or "HEAD"
        upper_id = upper_neck_part_id or self.cfg.upper_neck_part_id or "UPPER_NECK"
        lower_id = lower_neck_part_id or self.cfg.lower_neck_part_id or "LOWER_NECK"
        t1_id = self.cfg.t1_node_id or "T1"

        def _get(entity_id: str) -> Optional[pd.DataFrame]:
            """Exact key then partial match."""
            if entity_id in self.th:
                return self.th[entity_id]
            eid_up = entity_id.upper()
            for k, v in self.th.items():
                if eid_up in k.upper():
                    return v
            logger.warning("Entity '%s' not found in TH data. Available: %s",
                           entity_id, list(self.th.keys())[:10])
            return None

        metrics: dict = {
            "run_id": self.cfg.run_id,
            "activated_groups": ";".join(self.cfg.activated_groups),
            "t_start_ms": self.cfg.t_start_ms,
            "magnitude_pct": self.cfg.magnitude_pct,
        }

        # Head acceleration
        head_df = _get(head_id)
        if head_df is not None and "time" in head_df.columns:
            res_g = self.compute_resultant_g(head_df)
            t = head_df["time"]
            metrics["peak_head_accel_g"] = self.extract_peak_head_accel(res_g)
            metrics["HIC15"] = self.extract_hic15_fast(res_g, t)
            metrics["peak_g"] = metrics["peak_head_accel_g"]  # alias for reporter
        else:
            metrics["peak_head_accel_g"] = float("nan")
            metrics["HIC15"] = float("nan")
            metrics["peak_g"] = float("nan")

        # Head-T1 angle
        t1_df = _get(t1_id)
        if head_df is not None and t1_df is not None:
            angle_series = self.extract_head_t1_angle(head_df, t1_df)
            metrics["peak_head_t1_angle_deg"] = float(np.nanmax(np.abs(angle_series.values)))
            metrics["_head_t1_angle_series"] = angle_series
        else:
            metrics["peak_head_t1_angle_deg"] = float("nan")
            metrics["_head_t1_angle_series"] = None

        # Neck loads
        upper_df = _get(upper_id)
        lower_df = _get(lower_id)
        if upper_df is not None and lower_df is not None:
            neck = self.extract_neck_loads(upper_df, lower_df)
            metrics.update(neck)
        else:
            for k in ("upper_neck_Fx_N", "upper_neck_Fz_N", "upper_neck_My_Nm",
                      "lower_neck_Fx_N", "lower_neck_Fz_N", "lower_neck_My_Nm"):
                metrics[k] = float("nan")

        # Store raw time-history series for plotting
        metrics["_head_df"] = head_df
        metrics["_upper_neck_df"] = upper_df
        metrics["_lower_neck_df"] = lower_df

        return metrics


# ---------------------------------------------------------------------------
# SweepReporter
# ---------------------------------------------------------------------------

class SweepReporter:
    """
    Consolidates results from a parameter sweep and generates CSV + HTML reports.

    Parameters
    ----------
    results : list[dict]
        Each dict is the output of MetricsExtractor.extract_all(), optionally
        augmented with 'run_id' and 'config' keys.
    """

    _SCALAR_COLS = [
        "run_id",
        "activated_groups",
        "t_start_ms",
        "magnitude_pct",
        "peak_g",
        "HIC15",
        "peak_head_accel_g",
        "peak_head_t1_angle_deg",
        "upper_neck_Fx_N",
        "upper_neck_Fz_N",
        "upper_neck_My_Nm",
        "lower_neck_Fx_N",
        "lower_neck_Fz_N",
        "lower_neck_My_Nm",
    ]

    def __init__(self, results: list[dict]) -> None:
        self.results = results
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Build scalar summary table
    # ------------------------------------------------------------------

    def _build_df(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        rows = []
        for r in self.results:
            row = {col: r.get(col, float("nan")) for col in self._SCALAR_COLS}
            rows.append(row)
        self._df = pd.DataFrame(rows, columns=self._SCALAR_COLS)
        return self._df

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def to_csv(self, output_path: str) -> None:
        """Write consolidated scalar metrics to CSV."""
        df = self._build_df()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, float_format="%.4f")
        logger.info("CSV written: %s", out)

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def to_html_report(self, output_path: str) -> None:
        """
        Generate a self-contained HTML report with:
        - Summary table with min/max highlighting per metric column.
        - Time-history plots for key signals (embedded as base64 PNG).
        """
        df = self._build_df()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        table_html = self._build_table_html(df)
        plots_html = self._build_plots_html()

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OpenRadioss Rear-Impact Sweep Report</title>
<style>
  body {{
    font-family: "Segoe UI", Arial, sans-serif;
    background: #0d1117;
    color: #c9d1d9;
    margin: 0;
    padding: 24px;
  }}
  h1 {{
    color: #58a6ff;
    border-bottom: 2px solid #30363d;
    padding-bottom: 8px;
  }}
  h2 {{
    color: #79c0ff;
    margin-top: 36px;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 13px;
    margin-top: 12px;
  }}
  th {{
    background: #161b22;
    color: #58a6ff;
    padding: 8px 12px;
    text-align: left;
    border: 1px solid #30363d;
    white-space: nowrap;
  }}
  td {{
    padding: 6px 12px;
    border: 1px solid #30363d;
    white-space: nowrap;
  }}
  tr:nth-child(even) {{ background: #161b22; }}
  tr:hover {{ background: #1f2937; }}
  .cell-max {{ background: #3d1a1a; color: #ff7b72; font-weight: bold; }}
  .cell-min {{ background: #1a3d1a; color: #56d364; font-weight: bold; }}
  .plot-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
    gap: 20px;
    margin-top: 16px;
  }}
  .plot-card {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
  }}
  .plot-card h3 {{
    color: #79c0ff;
    margin: 0 0 10px 0;
    font-size: 14px;
  }}
  .plot-card img {{
    width: 100%;
    height: auto;
    border-radius: 4px;
  }}
  .footer {{
    margin-top: 40px;
    font-size: 11px;
    color: #484f58;
    border-top: 1px solid #30363d;
    padding-top: 10px;
  }}
</style>
</head>
<body>
<h1>OpenRadioss Rear-Impact Parameter Sweep</h1>
<p>VIVA+ HBM - Yoganandan 2000 sled setup | {len(self.results)} run(s)</p>

<h2>Summary Metrics</h2>
{table_html}

<h2>Time-History Plots</h2>
<div class="plot-grid">
{plots_html}
</div>

<div class="footer">
  Generated by postprocessor.py | {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S UTC")}
</div>
</body>
</html>"""

        out.write_text(html, encoding="utf-8")
        logger.info("HTML report written: %s", out)

    # ------------------------------------------------------------------
    # Internal: styled HTML table
    # ------------------------------------------------------------------

    def _build_table_html(self, df: pd.DataFrame) -> str:
        numeric_cols = [c for c in self._SCALAR_COLS
                        if c not in ("run_id", "activated_groups") and c in df.columns]

        # Pre-compute min/max index per numeric column
        col_min: dict[str, int] = {}
        col_max: dict[str, int] = {}
        for col in numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                col_min[col] = int(series.idxmin())
                col_max[col] = int(series.idxmax())

        headers = "".join(f"<th>{c}</th>" for c in self._SCALAR_COLS if c in df.columns)
        rows_html = ""
        for i, row in df.iterrows():
            cells = ""
            for col in self._SCALAR_COLS:
                if col not in df.columns:
                    continue
                val = row[col]
                cls = ""
                if col in col_max and i == col_max[col]:
                    cls = ' class="cell-max"'
                elif col in col_min and i == col_min[col]:
                    cls = ' class="cell-min"'
                if isinstance(val, float):
                    display = f"{val:.3f}" if not math.isnan(val) else "N/A"
                else:
                    display = str(val)
                cells += f"<td{cls}>{display}</td>"
            rows_html += f"<tr>{cells}</tr>\n"

        return f"<table><thead><tr>{headers}</tr></thead><tbody>{rows_html}</tbody></table>"

    # ------------------------------------------------------------------
    # Internal: time-history plots
    # ------------------------------------------------------------------

    def _build_plots_html(self) -> str:
        plot_specs = [
            ("Head Resultant Acceleration (g)", "_head_df", "resultant_g", "Time (s)", "Accel (g)"),
            ("Head-T1 Angle (deg)", "_head_t1_angle_series", None, "Time (s)", "Angle (deg)"),
            ("Upper Neck Fx (N)", "_upper_neck_df", "FX", "Time (s)", "Force (N)"),
            ("Upper Neck Fz (N)", "_upper_neck_df", "FZ", "Time (s)", "Force (N)"),
            ("Upper Neck My (N.m)", "_upper_neck_df", "MY", "Time (s)", "Moment (N.m)"),
            ("Lower Neck Fz (N)", "_lower_neck_df", "FZ", "Time (s)", "Force (N)"),
        ]

        cards = ""
        for title, data_key, col, xlabel, ylabel in plot_specs:
            b64 = self._make_plot(title, data_key, col, xlabel, ylabel)
            if b64:
                cards += (
                    f'<div class="plot-card"><h3>{title}</h3>'
                    f'<img src="data:image/png;base64,{b64}" alt="{title}"></div>\n'
                )
        return cards

    def _make_plot(
        self,
        title: str,
        data_key: str,
        col: Optional[str],
        xlabel: str,
        ylabel: str,
    ) -> Optional[str]:
        """Render a matplotlib plot for all runs and return base64 PNG string."""
        fig, ax = plt.subplots(figsize=(7, 3.5), dpi=100)
        ax.set_facecolor("#0d1117")
        fig.patch.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9", labelsize=9)
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#79c0ff")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, color="#30363d", linewidth=0.5)

        colors = plt.cm.tab10(np.linspace(0, 1, max(len(self.results), 1)))
        plotted = False

        for idx, r in enumerate(self.results):
            run_id = r.get("run_id", f"run_{idx}")
            color = colors[idx]

            if data_key == "_head_t1_angle_series":
                series = r.get("_head_t1_angle_series")
                if series is not None and len(series) > 0:
                    ax.plot(series.index, series.values, color=color, lw=1.2, label=run_id)
                    plotted = True
            else:
                df = r.get(data_key)
                if df is None or "time" not in df.columns:
                    continue
                if col == "resultant_g":
                    vals = MetricsExtractor.compute_resultant_g(df)
                    t = df["time"].values
                else:
                    col_map = {c.upper(): c for c in df.columns}
                    real_col = col_map.get(col.upper() if col else "")
                    if real_col is None:
                        continue
                    vals = df[real_col].values
                    t = df["time"].values
                ax.plot(t, vals, color=color, lw=1.2, label=run_id)
                plotted = True

        if not plotted:
            plt.close(fig)
            return None

        if len(self.results) > 1:
            legend = ax.legend(
                fontsize=8,
                framealpha=0.3,
                facecolor="#161b22",
                edgecolor="#30363d",
                labelcolor="#c9d1d9",
            )

        plt.tight_layout(pad=0.8)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")


# ---------------------------------------------------------------------------
# Convenience: run a single-file post-process from command line
# ---------------------------------------------------------------------------

def postprocess_single(
    th_file: str,
    run_config: RunConfig,
    output_dir: str = ".",
) -> dict:
    """
    Parse one T01 file, extract all metrics, return dict.
    Also writes CSV and HTML report to output_dir.
    """
    parser = THFileParser(th_file)
    th_data = parser.parse()

    extractor = MetricsExtractor(th_data, run_config)
    metrics = extractor.extract_all()

    reporter = SweepReporter([metrics])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reporter.to_csv(str(out / f"{run_config.run_id}_metrics.csv"))
    reporter.to_html_report(str(out / f"{run_config.run_id}_report.html"))

    return metrics


def postprocess_sweep(
    run_list: list[tuple[str, RunConfig]],
    output_dir: str = ".",
    report_name: str = "sweep_report",
) -> pd.DataFrame:
    """
    Run postprocessing across a parameter sweep.

    Parameters
    ----------
    run_list : list of (th_file_path, RunConfig)
    output_dir : destination folder for outputs
    report_name : base name for CSV and HTML files (no extension)

    Returns
    -------
    pd.DataFrame of scalar metrics for all runs.
    """
    all_metrics = []
    for th_file, cfg in run_list:
        logger.info("Processing run %s from %s", cfg.run_id, th_file)
        try:
            parser = THFileParser(th_file)
            th_data = parser.parse()
            extractor = MetricsExtractor(th_data, cfg)
            m = extractor.extract_all()
            all_metrics.append(m)
        except Exception as exc:
            logger.error("Failed on run %s: %s", cfg.run_id, exc)
            all_metrics.append({"run_id": cfg.run_id, **{k: float("nan") for k in SweepReporter._SCALAR_COLS if k != "run_id"}})

    reporter = SweepReporter(all_metrics)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reporter.to_csv(str(out / f"{report_name}.csv"))
    reporter.to_html_report(str(out / f"{report_name}.html"))

    return reporter._build_df()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser_cli = argparse.ArgumentParser(
        description="OpenRadioss T01 postprocessor for rear-impact biomechanics."
    )
    parser_cli.add_argument("th_file", help="Path to T01 time-history file")
    parser_cli.add_argument("--run-id", default="run_01", help="Run identifier")
    parser_cli.add_argument("--head-id", default="HEAD", help="Head entity ID/keyword")
    parser_cli.add_argument("--upper-neck-id", default="UPPER_NECK", help="Upper neck entity ID/keyword")
    parser_cli.add_argument("--lower-neck-id", default="LOWER_NECK", help="Lower neck entity ID/keyword")
    parser_cli.add_argument("--t1-id", default="T1", help="T1 vertebra node ID/keyword")
    parser_cli.add_argument("--t-start-ms", type=float, default=0.0, help="Pulse onset time (ms)")
    parser_cli.add_argument("--magnitude-pct", type=float, default=100.0, help="Pulse magnitude %%")
    parser_cli.add_argument("--groups", nargs="*", default=[], help="Activated muscle groups")
    parser_cli.add_argument("--output-dir", default=".", help="Output directory")
    parser_cli.add_argument("--list-entities", action="store_true", help="Print entity keys and exit")

    args = parser_cli.parse_args()

    cfg = RunConfig(
        run_id=args.run_id,
        activated_groups=args.groups or [],
        t_start_ms=args.t_start_ms,
        magnitude_pct=args.magnitude_pct,
        head_part_id=args.head_id,
        upper_neck_part_id=args.upper_neck_id,
        lower_neck_part_id=args.lower_neck_id,
        t1_node_id=args.t1_id,
    )

    p = THFileParser(args.th_file)
    data = p.parse()

    if args.list_entities:
        print("Entities found in", args.th_file)
        for k in data:
            df = data[k]
            print(f"  {k}: {list(df.columns)} | {len(df)} rows")
    else:
        metrics = postprocess_single(args.th_file, cfg, args.output_dir)
        # Print scalar metrics to stdout (exclude private keys)
        print(json.dumps(
            {k: (v if not isinstance(v, float) or not math.isnan(v) else None)
             for k, v in metrics.items() if not k.startswith("_")},
            indent=2,
        ))
