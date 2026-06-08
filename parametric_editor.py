"""
parametric_editor.py

Parametric editor for OpenRadioss .rad decks - Hill-type cervical muscle activation studies.

Modifies:
- Muscle activation onset time (T_start) via /FUNCT time-axis shift
- MVC magnitude scaling via Force field in /PROP/TYPE46 or via activation curve Y-scaling
- Baseline resting tone for non-activated muscle groups
- Crash pulse amplitude scaling via /IMPVEL or /IMPACC /FUNCT reference

Preserves:
- Native Hill-type force-length (fct_ID2) and force-velocity (fct_ID3) curve shapes
- Passive elastic curves (fct_ID4)
- Non-cervical muscle properties
- All structural (non-muscle) parts and properties

VIVA+ model reference:
- Neck muscle PIDs: left 205xxx-209xxx, right 255xxx-259xxx
- /PROP/TYPE46 (SPR_MUSCLE) or /PROP/SPR_MUSCLE for each part
- Activation curve referenced as fct_ID1 in the property card
- 8 control groups: SCM, STH, Scal, Trap, SCap, SCerv, CM_C4, CM_C6
"""

from __future__ import annotations

import copy
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants - VIVA+ cervical muscle PID ranges and group mappings
# ---------------------------------------------------------------------------

# Left-side PID prefix ranges (6-digit, starting with 20xxxx)
# Right-side counterparts = left PID + 50000
CERVICAL_PID_LEFT_RANGE = (205000, 210000)
CERVICAL_PID_RIGHT_OFFSET = 50000

# Muscle group to anatomical name fragments for PID-to-group assignment
# Derived from VIVA+ deck naming conventions
MUSCLE_GROUP_PATTERNS: Dict[str, List[str]] = {
    "SCM":    ["SCM", "Sternocleidomastoid"],
    "STH":    ["Sternohyoid", "Sternothyroid", "Omohyoid"],
    "Scal":   ["Scalenus-Anterior", "Scalenus-Medius", "Scalenus-Posterior",
               "Scalenus_Anterior", "Scalenus_Medius", "Scalenus_Posterior",
               "Scalenus"],
    "Trap":   ["Trapezius"],
    "SCap":   ["Splenius-Capitis", "Splenius_Capitis"],
    "SCerv":  ["Splenius-Cervicis", "Splenius_Cervicis"],
    "CM_C4":  ["Semispinalis-Capitis", "Longissimus-Capitis",
               "Semispinalis_Capitis", "Longissimus_Capitis"],
    "CM_C6":  ["Semispinalis-Cervicis", "Longissimus-Cervicis",
               "Multifidus", "Longus-Capitis", "Longus-Colli",
               "Semispinalis_Cervicis", "Longissimus_Cervicis",
               "Longus_Capitis", "Longus_Colli"],
}

ALL_CERVICAL_GROUPS = list(MUSCLE_GROUP_PATTERNS.keys())

# Default activation curve shape (normalized, t=0 at onset, rise 10ms, peak at 40ms)
# Y-values are activation fraction 0-1 (peak = 1.0, baseline handled separately)
DEFAULT_ACTIVATION_RISE_MS = 10.0   # ms
DEFAULT_ACTIVATION_DECAY_MS = 40.0  # ms


# ---------------------------------------------------------------------------
# RunConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """
    Parameters for a single parametric simulation run.

    Attributes
    ----------
    run_id : str
        Unique identifier for this run. Used as output subdirectory name.
    activated_groups : list[str]
        Subset of 8 cervical group names to actively contract.
        Must be a subset of: SCM, STH, Scal, Trap, SCap, SCerv, CM_C4, CM_C6.
        Groups not listed receive baseline_tone_pct as constant activation.
    t_start : float
        Activation onset time in seconds. Negative = pre-impact, positive = post-impact.
        Range: -0.060 to +0.100 s.
    magnitude_pct : float
        Peak activation as percent of MVC. Range: 0 to 150.
        150 = supramaximal (1.5x Fmax).
    baseline_tone_pct : float
        Resting tone for non-activated groups as percent of MVC. Default 3.5.
        Range: 0 to 10 (sanity clamp; physiological range is 2-5%).
    peak_g : float
        Crash pulse peak acceleration in g. Range: 5 to 15.
    notes : str
        Optional free-text annotation written into the deck comment header.
    """
    run_id: str
    activated_groups: List[str]
    t_start: float
    magnitude_pct: float
    baseline_tone_pct: float = 3.5
    peak_g: float = 10.0
    notes: str = ""

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Raise ValueError on out-of-range parameters."""
        if self.magnitude_pct < 0 or self.magnitude_pct > 150:
            raise ValueError(
                f"magnitude_pct must be in [0, 150], got {self.magnitude_pct}"
            )
        if self.t_start < -0.060 or self.t_start > 0.100:
            raise ValueError(
                f"t_start must be in [-0.060, +0.100] s, got {self.t_start}"
            )
        if self.peak_g < 5 or self.peak_g > 15:
            raise ValueError(
                f"peak_g must be in [5, 15] g, got {self.peak_g}"
            )
        unknown = set(self.activated_groups) - set(ALL_CERVICAL_GROUPS)
        if unknown:
            raise ValueError(
                f"Unknown muscle groups: {unknown}. "
                f"Valid groups: {ALL_CERVICAL_GROUPS}"
            )
        if self.baseline_tone_pct < 0 or self.baseline_tone_pct > 10:
            raise ValueError(
                f"baseline_tone_pct must be in [0, 10], got {self.baseline_tone_pct}"
            )


# ---------------------------------------------------------------------------
# Radioss .rad format helpers - field-level parsing
# ---------------------------------------------------------------------------

RAD_LINE_WIDTH = 100
RAD_FIELD_WIDTH = 10


def _field(line: str, idx: int) -> str:
    """
    Return the raw string for 0-indexed field idx in a 100-char Radioss line.
    Field idx=0 => columns 0-9, idx=1 => 10-19, etc.
    """
    start = idx * RAD_FIELD_WIDTH
    end = start + RAD_FIELD_WIDTH
    return line[start:end] if len(line) >= end else line[start:].ljust(RAD_FIELD_WIDTH)


def _set_field(line: str, idx: int, value: str) -> str:
    """
    Return a new line with field idx replaced by value (right-justified, 10 chars).
    Pads or truncates line to 100 chars as needed.
    """
    line = line.ljust(RAD_LINE_WIDTH)
    start = idx * RAD_FIELD_WIDTH
    formatted = value[:RAD_FIELD_WIDTH].rjust(RAD_FIELD_WIDTH)
    return line[:start] + formatted + line[start + RAD_FIELD_WIDTH:]


def _fmt_float(v: float) -> str:
    """Format a float to fit in a 10-char Radioss field (scientific notation)."""
    s = f"{v:.5E}"
    if len(s) > 10:
        s = f"{v:.3E}"
    return s[:10]


def _fmt_int(v: int) -> str:
    return str(v)[:10]


def _parse_float(s: str) -> float:
    s = s.strip()
    if not s:
        return 0.0
    return float(s)


def _parse_int(s: str) -> int:
    s = s.strip()
    if not s:
        return 0
    return int(s)


def _is_comment(line: str) -> bool:
    stripped = line.lstrip()
    if not stripped:
        return False
    # # and $ are comments; but #RADIOSS STARTER, #enddata, #include are not
    if stripped.startswith("$"):
        return True
    if stripped.startswith("#"):
        lower = stripped.lower()
        if lower.startswith("#radioss") or lower.startswith("#enddata") or lower.startswith("#include"):
            return False
        return True
    return False


def _is_keyword(line: str) -> bool:
    """True if line starts a keyword block (starts with '/' but not '//')."""
    stripped = line.lstrip()
    return stripped.startswith("/") and not stripped.startswith("//")


# ---------------------------------------------------------------------------
# Block data structures
# ---------------------------------------------------------------------------

@dataclass
class RadiossBlock:
    """
    Represents a single keyword block in the .rad deck.

    Attributes
    ----------
    keyword : str
        The keyword header line, e.g. '/FUNCT/101'.
    raw_lines : list[str]
        All lines belonging to this block including the keyword header line,
        title, data lines, and blank/comment lines within the block.
    line_start : int
        0-based index of the keyword header line in the original file.
    """
    keyword: str
    raw_lines: List[str]
    line_start: int

    @property
    def keyword_upper(self) -> str:
        return self.keyword.strip().upper()

    def keyword_parts(self) -> List[str]:
        """Split keyword line on '/' returning non-empty parts."""
        return [p for p in self.keyword.strip().split("/") if p]

    def block_id(self) -> Optional[int]:
        """Return the numeric ID field from the keyword line, or None."""
        parts = self.keyword_parts()
        for p in parts[1:]:
            try:
                return int(p)
            except ValueError:
                continue
        return None

    def data_lines(self) -> List[Tuple[int, str]]:
        """
        Return (index_in_raw_lines, line) for all non-comment, non-keyword lines
        after the keyword header line. Index 0 = first line after keyword = title line.
        """
        result = []
        for i, line in enumerate(self.raw_lines[1:], start=1):
            if not _is_comment(line):
                result.append((i, line))
        return result


# ---------------------------------------------------------------------------
# DeckParser - reads and writes OpenRadioss .rad files
# ---------------------------------------------------------------------------

class DeckParser:
    """
    Parses an OpenRadioss .rad starter deck into blocks and allows
    targeted in-place modifications before writing back to disk.

    Usage
    -----
        parser = DeckParser.from_file("model.rad")
        parser2 = parser.deep_copy()
        # ... modify parser2 blocks ...
        parser2.write("modified.rad")
    """

    def __init__(self, lines: List[str]) -> None:
        self._raw_lines: List[str] = lines
        self._blocks: List[RadiossBlock] = []
        self._parse()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str) -> "DeckParser":
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        # Normalize line endings but preserve trailing content
        lines = [ln.rstrip("\n").rstrip("\r") for ln in lines]
        return cls(lines)

    @classmethod
    def from_string(cls, text: str) -> "DeckParser":
        lines = text.splitlines()
        return cls(lines)

    def deep_copy(self) -> "DeckParser":
        new = DeckParser.__new__(DeckParser)
        new._raw_lines = copy.deepcopy(self._raw_lines)
        new._blocks = copy.deepcopy(self._blocks)
        return new

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self) -> None:
        """
        Split raw lines into keyword blocks.
        Each block begins at a '/' keyword line and ends just before the next.
        Preamble lines (before first keyword) are stored as a pseudo-block.
        """
        self._blocks = []
        current_keyword = None
        current_lines: List[str] = []
        current_start = 0

        for i, line in enumerate(self._raw_lines):
            if _is_keyword(line):
                if current_keyword is not None or current_lines:
                    self._blocks.append(
                        RadiossBlock(
                            keyword=current_keyword or "",
                            raw_lines=current_lines,
                            line_start=current_start,
                        )
                    )
                current_keyword = line
                current_lines = [line]
                current_start = i
            else:
                if current_keyword is None and not self._blocks:
                    # preamble
                    if not current_lines:
                        current_start = i
                        current_keyword = ""
                    current_lines.append(line)
                else:
                    current_lines.append(line)

        if current_lines:
            self._blocks.append(
                RadiossBlock(
                    keyword=current_keyword or "",
                    raw_lines=current_lines,
                    line_start=current_start,
                )
            )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_blocks_by_keyword(self, keyword_fragment: str) -> List[RadiossBlock]:
        """
        Return all blocks whose keyword line contains keyword_fragment (case-insensitive).
        E.g. 'FUNCT' matches '/FUNCT/101', '/FUNCT_SMOOTH/22', etc.
        """
        fragment_upper = keyword_fragment.upper()
        return [b for b in self._blocks if fragment_upper in b.keyword_upper]

    def get_block_by_id(self, keyword_fragment: str, block_id: int) -> Optional[RadiossBlock]:
        """Return the first block matching keyword_fragment with the given numeric ID."""
        for b in self.get_blocks_by_keyword(keyword_fragment):
            if b.block_id() == block_id:
                return b
        return None

    def get_all_part_blocks(self) -> List[RadiossBlock]:
        return self.get_blocks_by_keyword("/PART")

    def get_all_prop_blocks(self) -> List[RadiossBlock]:
        return (
            self.get_blocks_by_keyword("/PROP/TYPE46")
            + self.get_blocks_by_keyword("/PROP/SPR_MUSCLE")
        )

    def get_all_funct_blocks(self) -> List[RadiossBlock]:
        return (
            self.get_blocks_by_keyword("/FUNCT")
            + self.get_blocks_by_keyword("/FUNCT_SMOOTH")
        )

    # ------------------------------------------------------------------
    # Block-level modification
    # ------------------------------------------------------------------

    def set_block_data_field(
        self,
        block: RadiossBlock,
        data_line_idx: int,
        field_idx: int,
        value: str,
    ) -> None:
        """
        Modify field_idx (0-based) on data_line_idx (0-based, counting only
        non-comment lines after the keyword header) within block.raw_lines.

        Also updates self._raw_lines to keep them in sync.
        """
        data_lines = block.data_lines()
        if data_line_idx >= len(data_lines):
            raise IndexError(
                f"Block {block.keyword}: data_line_idx {data_line_idx} out of range "
                f"(block has {len(data_lines)} data lines)"
            )
        raw_idx, current_line = data_lines[data_line_idx]
        new_line = _set_field(current_line, field_idx, value)
        block.raw_lines[raw_idx] = new_line
        # Sync back to _raw_lines
        self._raw_lines[block.line_start + raw_idx] = new_line

    def replace_block_line(self, block: RadiossBlock, raw_idx: int, new_line: str) -> None:
        """Replace raw_lines[raw_idx] in block, keeping _raw_lines in sync."""
        block.raw_lines[raw_idx] = new_line
        self._raw_lines[block.line_start + raw_idx] = new_line

    # ------------------------------------------------------------------
    # /FUNCT-level helpers
    # ------------------------------------------------------------------

    def get_funct_xy_data(self, funct_id: int) -> Tuple[Optional[RadiossBlock], List[Tuple[float, float]]]:
        """
        Return (block, [(x, y), ...]) for a /FUNCT block with the given ID.
        Data lines after the scale/shift line (data_line_idx >= 1) are X-Y pairs.
        Returns (None, []) if not found.

        Parsing strategy:
        - For XY data lines (i >= 2), whitespace tokenization is used as a robust
          fallback because real deck files may not pad all values to strict 10-char
          field boundaries (especially when values like -0.050 span a boundary).
        - Strict 10-char field parsing is attempted first; if it fails the line is
          re-tokenised on whitespace.
        """
        block = self.get_block_by_id("FUNCT", funct_id)
        if block is None:
            return None, []

        data_lines = block.data_lines()
        # data_lines[0] = title line (text, no fields)
        # data_lines[1] = Ascalex Fscaley Ashiftx Fshifty (scale/shift line)
        # data_lines[2..] = X Y pairs

        xy: List[Tuple[float, float]] = []
        for i, (raw_idx, line) in enumerate(data_lines):
            if i < 2:
                continue  # title + scale line

            # Try strict field layout first
            parsed = False
            x_str = _field(line, 0)
            y_str = _field(line, 1)
            try:
                x = _parse_float(x_str)
                y = _parse_float(y_str)
                xy.append((x, y))
                parsed = True
            except ValueError:
                pass

            # Fallback: whitespace tokenisation
            if not parsed:
                tokens = line.split()
                if len(tokens) >= 2:
                    try:
                        x = float(tokens[0])
                        y = float(tokens[1])
                        xy.append((x, y))
                    except ValueError:
                        continue

        return block, xy

    def set_funct_scale_shift(
        self,
        block: RadiossBlock,
        ascalex: Optional[float] = None,
        fscaley: Optional[float] = None,
        ashiftx: Optional[float] = None,
        fshifty: Optional[float] = None,
    ) -> None:
        """
        Modify the Ascalex / Fscaley / Ashiftx / Fshifty fields on the
        scale/shift line (data_line_idx=1) of a /FUNCT block.
        """
        # data_line_idx=0 is title (plain text), idx=1 is the scale/shift line
        data_lines = block.data_lines()
        if len(data_lines) < 2:
            return
        raw_idx, current_line = data_lines[1]
        line = current_line.ljust(RAD_LINE_WIDTH)

        if ascalex is not None:
            line = _set_field(line, 0, _fmt_float(ascalex))
        if fscaley is not None:
            line = _set_field(line, 1, _fmt_float(fscaley))
        if ashiftx is not None:
            line = _set_field(line, 2, _fmt_float(ashiftx))
        if fshifty is not None:
            line = _set_field(line, 3, _fmt_float(fshifty))

        self.replace_block_line(block, raw_idx, line)

    def scale_funct_y_values(self, block: RadiossBlock, scale: float) -> None:
        """
        Multiply every Y value in a /FUNCT block's data points by scale.
        Preserves X values unchanged. Title and scale/shift lines are not touched.
        """
        data_lines = block.data_lines()
        for i, (raw_idx, line) in enumerate(data_lines):
            if i < 2:
                continue
            x_str = _field(line, 0)
            y_str = _field(line, 1)
            try:
                y = _parse_float(y_str)
            except ValueError:
                continue
            new_y = y * scale
            new_line = line.ljust(RAD_LINE_WIDTH)
            new_line = _set_field(new_line, 0, x_str.strip().rjust(RAD_FIELD_WIDTH))
            new_line = _set_field(new_line, 1, _fmt_float(new_y))
            self.replace_block_line(block, raw_idx, new_line)

    def get_funct_peak_y(self, funct_id: int) -> Optional[float]:
        """Return the maximum absolute Y value in a /FUNCT's data points."""
        _, xy = self.get_funct_xy_data(funct_id)
        if not xy:
            return None
        return max(abs(y) for _, y in xy)

    def replace_funct_with_constant(self, block: RadiossBlock, constant_value: float) -> None:
        """
        Replace all X-Y data lines in a /FUNCT block with a flat constant:
        two points at t=-1e9 and t=1e9 both equal to constant_value.
        This produces a constant activation level over all time.
        Preserves title and scale/shift lines.
        """
        data_lines = block.data_lines()

        # Remove existing XY data lines from raw_lines (keep title + scale line)
        # We'll mark indices to replace, then rebuild
        xy_raw_indices = [raw_idx for i, (raw_idx, _) in enumerate(data_lines) if i >= 2]

        if not xy_raw_indices:
            return

        new_line_a = (
            _fmt_float(-1.0e9).rjust(RAD_FIELD_WIDTH)
            + _fmt_float(constant_value).rjust(RAD_FIELD_WIDTH)
        ).ljust(RAD_LINE_WIDTH)
        new_line_b = (
            _fmt_float(1.0e9).rjust(RAD_FIELD_WIDTH)
            + _fmt_float(constant_value).rjust(RAD_FIELD_WIDTH)
        ).ljust(RAD_LINE_WIDTH)

        # Replace first two XY indices with the constant pair; blank the rest
        for k, raw_idx in enumerate(xy_raw_indices):
            if k == 0:
                new_line = new_line_a
            elif k == 1:
                new_line = new_line_b
            else:
                new_line = ""
            block.raw_lines[raw_idx] = new_line
            self._raw_lines[block.line_start + raw_idx] = new_line

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, path: str) -> None:
        """Write the current state of the deck to path."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for line in self._raw_lines:
                fh.write(line + "\n")

    def to_string(self) -> str:
        return "\n".join(self._raw_lines)


# ---------------------------------------------------------------------------
# MuscleMapper - maps part IDs to groups and activation /FUNCT IDs
# ---------------------------------------------------------------------------

class MuscleMapper:
    """
    Builds the mapping between cervical muscle parts, their groups,
    and their activation /FUNCT curve IDs by parsing the deck.

    This class handles both:
    - /PROP/TYPE46 (SPR_MUSCLE): fct_ID1 is the activation curve
    - Fallback: /PROP/TYPE4 (generic spring) with activation curve in field 1

    For VIVA+-style LS-DYNA .k decks translated to .rad:
    - Part PID is used to identify the part
    - The part's property ID references a /PROP block
    - The /PROP block's fct_ID1 field references the activation /FUNCT

    Attributes
    ----------
    part_to_group : dict[int, str]
        Maps cervical part IDs to their group name (SCM, STH, etc.)
    part_to_funct : dict[int, int]
        Maps part IDs to their activation /FUNCT curve ID
    part_to_prop : dict[int, int]
        Maps part IDs to their property block ID
    cervical_pids : set[int]
        All cervical muscle part IDs found in the deck
    """

    def __init__(self, parser: DeckParser) -> None:
        self.part_to_group: Dict[int, str] = {}
        self.part_to_funct: Dict[int, int] = {}
        self.part_to_prop: Dict[int, int] = {}
        self.part_to_name: Dict[int, str] = {}
        self.cervical_pids: set = set()
        self._build(parser)

    def _build(self, parser: DeckParser) -> None:
        """Parse the deck to build all mappings."""
        # Step 1: identify all /PART blocks and extract PID + title
        all_parts = parser.get_all_part_blocks()
        for part_block in all_parts:
            pid = part_block.block_id()
            if pid is None:
                continue

            data_lines = part_block.data_lines()
            # data_lines[0] = title (part name)
            title = data_lines[0][1].strip() if data_lines else ""
            self.part_to_name[pid] = title

            # Check if this is a cervical muscle part
            if not self._is_cervical_pid(pid) and not self._name_is_cervical(title):
                continue

            self.cervical_pids.add(pid)

            # Assign to group based on name
            group = self._group_from_name(title)
            if group:
                self.part_to_group[pid] = group

            # data_lines[1] should have: PID SECID MID GRNOD_ID (or similar)
            # For /PART: line 2 contains MID SECID fields or PROP_ID
            # We'll look for a referenced prop ID - for VIVA+ it's SECID = PROPID
            if len(data_lines) > 1:
                _, prop_line = data_lines[1]
                # Field 0 = MID or PID, field 1 = SECID/PROPID
                try:
                    prop_id = _parse_int(_field(prop_line, 1))
                    if prop_id > 0:
                        self.part_to_prop[pid] = prop_id
                except (ValueError, IndexError):
                    pass

        # Step 2: for each cervical part, resolve prop -> funct
        prop_blocks = parser.get_all_prop_blocks()
        prop_funct_map: Dict[int, int] = {}

        for prop_block in prop_blocks:
            prop_id = prop_block.block_id()
            if prop_id is None:
                continue
            funct_id = self._extract_activation_funct_from_prop(prop_block)
            if funct_id is not None:
                prop_funct_map[prop_id] = funct_id

        for pid in self.cervical_pids:
            prop_id = self.part_to_prop.get(pid)
            if prop_id and prop_id in prop_funct_map:
                self.part_to_funct[pid] = prop_funct_map[prop_id]

    def _is_cervical_pid(self, pid: int) -> bool:
        """Check if PID falls in the known VIVA+ cervical muscle range."""
        lo, hi = CERVICAL_PID_LEFT_RANGE
        right_lo = lo + CERVICAL_PID_RIGHT_OFFSET
        right_hi = hi + CERVICAL_PID_RIGHT_OFFSET
        return (lo <= pid < hi) or (right_lo <= pid < right_hi)

    def _name_is_cervical(self, name: str) -> bool:
        """Return True if name contains any known cervical muscle fragment."""
        name_upper = name.upper()
        for patterns in MUSCLE_GROUP_PATTERNS.values():
            for p in patterns:
                if p.upper() in name_upper:
                    return True
        return False

    def _group_from_name(self, name: str) -> Optional[str]:
        """Return the muscle group for a part name, or None if unrecognized."""
        name_upper = name.upper()
        for group, patterns in MUSCLE_GROUP_PATTERNS.items():
            for p in patterns:
                if p.upper() in name_upper:
                    return group
        return None

    def _extract_activation_funct_from_prop(self, prop_block: RadiossBlock) -> Optional[int]:
        """
        Extract the activation /FUNCT ID (fct_ID1) from a /PROP/TYPE46 block.

        /PROP/TYPE46 data layout (after title):
        Line 0 (title): free text
        Line 1 (data line 1): Mass Stiffness Vel_max Force Xk fct_ID1 fct_ID2 fct_ID3 fct_ID4 ...

        fct_ID1 is the 6th whitespace-token (index 5) on that line.
        We use whitespace tokenisation as a fallback because free-format Radioss
        decks may not pad all fields to exactly 10 chars.
        Strict 10-char field parsing is tried first; whitespace fallback second.
        """
        data_lines = prop_block.data_lines()
        # data_lines[0] = title, data_lines[1] = first real data line
        if len(data_lines) < 2:
            return None
        _, data_line = data_lines[1]

        # -- Strategy 1: strict 10-char field layout (field index 5) --
        try:
            fct_id = _parse_int(_field(data_line, 5))
            if fct_id > 0:
                return fct_id
        except (ValueError, IndexError):
            pass

        # -- Strategy 2: whitespace-tokenised (handles loose formatting) --
        tokens = data_line.split()
        if len(tokens) > 5:
            try:
                fct_id = int(tokens[5])
                if fct_id > 0:
                    return fct_id
            except ValueError:
                pass

        # -- Strategy 3: scan all tokens for a plausible /FUNCT ID (>100, integer) --
        # This handles decks where the field count differs from the canonical layout
        for tok in tokens:
            try:
                v = int(tok)
                if v > 100:
                    return v
            except ValueError:
                continue

        return None

    def pids_for_group(self, group: str) -> List[int]:
        """Return all part IDs belonging to the given muscle group."""
        return [pid for pid, g in self.part_to_group.items() if g == group]

    def funct_ids_for_group(self, group: str) -> List[int]:
        """Return all unique activation /FUNCT IDs for parts in a group."""
        pids = self.pids_for_group(group)
        funct_ids = list({self.part_to_funct[p] for p in pids if p in self.part_to_funct})
        return funct_ids


# ---------------------------------------------------------------------------
# ParametricEditor - main orchestrator
# ---------------------------------------------------------------------------

class ParametricEditor:
    """
    Applies parametric run configurations to an OpenRadioss .rad deck.

    Responsibilities
    ----------------
    - Modify activation onset (T_start) and magnitude for activated groups
    - Set baseline resting tone for non-activated cervical groups
    - Leave non-cervical muscles and all structural parts completely unchanged
    - Scale the crash pulse amplitude (/IMPVEL or /IMPACC)
    - Return a modified DeckParser ready to write to disk

    Parameters
    ----------
    deck_parser : DeckParser
        Parsed representation of the base .rad deck.
    muscle_mapper : MuscleMapper
        Pre-built mapping of part IDs to groups and /FUNCT IDs.
    crash_pulse_funct_id : int, optional
        The /FUNCT ID that drives the crash pulse velocity/acceleration profile.
        If None, the editor will attempt to auto-detect it from /IMPVEL or /IMPACC.
    crash_pulse_current_peak_g : float, optional
        Known peak g value of the crash pulse /FUNCT as-stored in the deck.
        If None, the editor will compute it from the /FUNCT Y values.
    """

    def __init__(
        self,
        deck_parser: DeckParser,
        muscle_mapper: MuscleMapper,
        crash_pulse_funct_id: Optional[int] = None,
        crash_pulse_current_peak_g: Optional[float] = None,
    ) -> None:
        self._base_parser = deck_parser
        self._mapper = muscle_mapper
        self._crash_pulse_funct_id = crash_pulse_funct_id
        self._crash_pulse_current_peak_g = crash_pulse_current_peak_g

        # Auto-detect crash pulse if not provided
        if self._crash_pulse_funct_id is None:
            self._crash_pulse_funct_id = self._detect_crash_pulse_funct(deck_parser)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_run_config(self, config: RunConfig) -> DeckParser:
        """
        Apply a RunConfig to a deep copy of the base deck.

        Steps
        -----
        a. Activated cervical groups:
           - Shift T_start on their activation /FUNCT (Ashiftx)
           - Scale peak magnitude (Fscaley or Y values)
           - Baseline = config.baseline_tone_pct (minimum Y floor)
        b. Non-activated cervical groups:
           - Set constant activation = config.baseline_tone_pct / 100.0
        c. Non-cervical muscles: unchanged
        d. Crash pulse: scale /FUNCT Y values to reach config.peak_g

        Returns
        -------
        DeckParser
            Modified deck, ready to write.
        """
        parser = self._base_parser.deep_copy()

        activated_set = set(config.activated_groups)
        baseline_fraction = config.baseline_tone_pct / 100.0
        magnitude_fraction = config.magnitude_pct / 100.0

        # (a) + (b) Process all cervical groups
        for group in ALL_CERVICAL_GROUPS:
            funct_ids = self._mapper.funct_ids_for_group(group)
            if not funct_ids:
                continue

            if group in activated_set:
                # Activated: set T_start and magnitude
                for funct_id in funct_ids:
                    self._apply_activation(
                        parser=parser,
                        funct_id=funct_id,
                        t_start=config.t_start,
                        magnitude_fraction=magnitude_fraction,
                        baseline_fraction=baseline_fraction,
                    )
            else:
                # Non-activated: set constant baseline tone
                for funct_id in funct_ids:
                    self._apply_baseline_tone(
                        parser=parser,
                        funct_id=funct_id,
                        baseline_fraction=baseline_fraction,
                    )

        # (d) Crash pulse scaling
        self._apply_crash_pulse_scaling(parser, config.peak_g)

        # Add run info as comment at top of deck
        self._inject_run_header(parser, config)

        return parser

    def generate_batch(
        self,
        configs: List[RunConfig],
        output_dir: str,
        base_deck_path: Optional[str] = None,
    ) -> List[str]:
        """
        Apply each RunConfig and write a modified Starter deck per run.

        OpenRadioss requires a Starter file named ``<root>_0000.rad`` plus a
        matching Engine file ``<root>_0001.rad``. When ``base_deck_path`` is
        provided and follows that convention, each run directory receives:

        - ``<root>_0000.rad`` : the parametrically modified Starter deck
        - ``<root>_0001.rad`` : the Engine deck copied unchanged from the base
          deck's directory (run controls / output requests are model-defined and
          must not be altered by the muscle-activation sweep)

        If ``base_deck_path`` is omitted (legacy behaviour) each run is written
        as a single ``model.rad`` for callers that supply their own solver
        invocation.

        Parameters
        ----------
        configs : list[RunConfig]
            List of parametric configurations to run.
        output_dir : str
            Root directory for output. Created if it does not exist.
        base_deck_path : str, optional
            Path to the base Starter deck (``<root>_0000.rad`` or ``<root>.rad``).
            Used to derive the solver root name and locate the sibling Engine file.

        Returns
        -------
        list[str]
            Absolute paths to the written Starter deck files.
        """
        output_paths = []
        output_root = Path(output_dir).resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        # Derive the OpenRadioss root name and locate the Engine (_0001) file.
        root_name = "model"
        engine_src: Optional[Path] = None
        if base_deck_path:
            base = Path(base_deck_path)
            stem = base.name
            if stem.endswith("_0000.rad"):
                root_name = stem[: -len("_0000.rad")]
            elif stem.endswith(".rad"):
                root_name = stem[: -len(".rad")]
            candidate = base.parent / f"{root_name}_0001.rad"
            if candidate.exists():
                engine_src = candidate
            else:
                logger.warning(
                    "No Engine deck found at %s - runs will be Starter-only and "
                    "produce no time-history (T01) output.",
                    candidate,
                )

        for config in configs:
            modified = self.apply_run_config(config)
            run_dir = output_root / f"run_{config.run_id}"
            run_dir.mkdir(parents=True, exist_ok=True)
            if base_deck_path:
                out_path = str(run_dir / f"{root_name}_0000.rad")
            else:
                out_path = str(run_dir / "model.rad")
            modified.write(out_path)
            if engine_src is not None:
                shutil.copyfile(engine_src, run_dir / f"{root_name}_0001.rad")
            output_paths.append(out_path)

        return output_paths

    # ------------------------------------------------------------------
    # Private: activation modification
    # ------------------------------------------------------------------

    def _apply_activation(
        self,
        parser: DeckParser,
        funct_id: int,
        t_start: float,
        magnitude_fraction: float,
        baseline_fraction: float,
    ) -> None:
        """
        Modify a single activation /FUNCT for an activated muscle group:

        1. Read the current curve data to understand its shape
        2. Use Ashiftx to shift the entire time axis so that activation onset = t_start
           (Ashiftx = shift applied BEFORE lookup: effective_t = t - Ashiftx)
           To move onset from t_onset_orig to t_start: Ashiftx = t_onset_orig - t_start
        3. Scale Y values so that:
           - peak = magnitude_fraction (MVC fraction)
           - minimum (resting) = baseline_fraction
           If magnitude_fraction == baseline_fraction, produces a constant curve.

        Hill curve shape (rise/decay) is preserved by using the Ashiftx shift
        rather than re-parameterizing individual X-Y points.
        """
        block, xy = parser.get_funct_xy_data(funct_id)
        if block is None or not xy:
            return

        # Find current onset time: x-value where y first exceeds baseline
        current_onset = self._find_onset_time(xy, threshold=baseline_fraction + 0.01)

        # Compute required Ashiftx:
        # OpenRadioss applies: y_eff = Fscaley * f((t - Ashiftx) / Ascalex) + Fshifty
        # So the internal x-axis lookup is t - Ashiftx.
        # Current onset in internal coords = current_onset.
        # We want the onset to appear at t = t_start in simulation time.
        # => t_start - Ashiftx = current_onset
        # => Ashiftx = t_start - current_onset
        ashiftx = t_start - current_onset

        # Compute Y scale: map current [0, 1] range to [baseline_fraction, magnitude_fraction]
        current_peak = max(y for _, y in xy) if xy else 1.0
        current_min = min(y for _, y in xy) if xy else 0.0

        if current_peak <= 0 or abs(current_peak - current_min) < 1e-6:
            # Degenerate curve: just replace with constant
            logger.warning(
                "Degenerate activation curve in block %s (peak=%.6f, min=%.6f): "
                "falling back to constant %.4f. Check VIVA+ deck integrity.",
                block.keyword, current_peak, current_min, magnitude_fraction
            )
            parser.replace_funct_with_constant(block, magnitude_fraction)
            parser.set_funct_scale_shift(block, ashiftx=ashiftx)
            return

        # Normalize Y to [0, 1] then rescale to [baseline_fraction, magnitude_fraction]
        # Y_new = baseline_fraction + (Y_current - current_min) / (current_peak - current_min)
        #                                * (magnitude_fraction - baseline_fraction)
        # This is equivalent to:
        # Fscaley = (magnitude_fraction - baseline_fraction) / (current_peak - current_min)
        # Fshifty = baseline_fraction - Fscaley * current_min
        current_range = current_peak - current_min
        target_range = magnitude_fraction - baseline_fraction

        fscaley = target_range / current_range
        fshifty = baseline_fraction - fscaley * current_min

        parser.set_funct_scale_shift(
            block,
            ashiftx=ashiftx,
            fscaley=fscaley,
            fshifty=fshifty,
        )

    def _apply_baseline_tone(
        self,
        parser: DeckParser,
        funct_id: int,
        baseline_fraction: float,
    ) -> None:
        """
        Set a non-activated muscle group's activation to a constant baseline.
        Replaces the /FUNCT data with a flat two-point constant curve.
        """
        block, _ = parser.get_funct_xy_data(funct_id)
        if block is None:
            return
        parser.replace_funct_with_constant(block, baseline_fraction)
        # Reset any scale/shift factors to identity
        parser.set_funct_scale_shift(block, ascalex=1.0, fscaley=1.0, ashiftx=0.0, fshifty=0.0)

    def _find_onset_time(
        self, xy: List[Tuple[float, float]], threshold: float = 0.05
    ) -> float:
        """
        Find the time (x-value) where activation first exceeds threshold.
        Falls back to the x-value of the first data point if no threshold crossing found.
        """
        if not xy:
            return 0.0
        for x, y in xy:
            if y >= threshold:
                return x
        return xy[0][0]

    # ------------------------------------------------------------------
    # Private: crash pulse scaling
    # ------------------------------------------------------------------

    def _apply_crash_pulse_scaling(self, parser: DeckParser, target_peak_g: float) -> None:
        """
        Scale the crash pulse /FUNCT so that its peak absolute Y value
        corresponds to target_peak_g.

        Approach:
        1. Get the crash pulse /FUNCT ID (from _crash_pulse_funct_id)
        2. Compute current peak Y (in g or equivalent units)
        3. Scale all Y values by target_peak_g / current_peak
        """
        if self._crash_pulse_funct_id is None:
            return

        funct_id = self._crash_pulse_funct_id
        block, xy = parser.get_funct_xy_data(funct_id)
        if block is None or not xy:
            return

        # Determine current peak (from pre-known value or compute from data)
        if self._crash_pulse_current_peak_g is not None:
            current_peak = self._crash_pulse_current_peak_g
        else:
            raw_peak = max(abs(y) for _, y in xy)
            # Account for any existing Fscaley already applied to this FUNCT block
            existing_fscaley = 1.0
            data_lines = block.data_lines()
            if len(data_lines) >= 2:
                _, scale_line = data_lines[1]
                try:
                    existing_fscaley = _parse_float(_field(scale_line, 1))
                except ValueError:
                    existing_fscaley = 1.0
                if existing_fscaley == 0.0:
                    existing_fscaley = 1.0
            current_peak = existing_fscaley * raw_peak

        if current_peak <= 0:
            return

        scale = target_peak_g / current_peak

        # Apply scale via Fscaley on the /FUNCT scale line
        # This is cleaner than modifying all Y data points
        data_lines = block.data_lines()
        if len(data_lines) >= 2:
            _, scale_line = data_lines[1]
            try:
                existing_fscaley = _parse_float(_field(scale_line, 1))
            except ValueError:
                existing_fscaley = 1.0
            if existing_fscaley == 0.0:
                existing_fscaley = 1.0
            new_fscaley = existing_fscaley * scale
            parser.set_funct_scale_shift(block, fscaley=new_fscaley)

    def _detect_crash_pulse_funct(self, parser: DeckParser) -> Optional[int]:
        """
        Auto-detect the crash pulse /FUNCT ID by scanning /IMPVEL and /IMPACC blocks.
        Returns the first fct_IDT found, or None.
        """
        for kw in ("/IMPVEL", "/IMPACC"):
            blocks = parser.get_blocks_by_keyword(kw)
            for block in blocks:
                data_lines = block.data_lines()
                # /IMPVEL line 1 (data_line_idx=1): fct_IDT Dir Skew_ID sens_ID grnd_ID frame_ID icoor
                # fct_IDT is field 0
                if len(data_lines) >= 2:
                    _, data_line = data_lines[1]
                    try:
                        fct_id = _parse_int(_field(data_line, 0))
                        if fct_id > 0:
                            return fct_id
                    except (ValueError, IndexError):
                        pass
        return None

    # ------------------------------------------------------------------
    # Private: helper for activation /FUNCT resolution
    # ------------------------------------------------------------------

    def _find_activation_funct_for_part(self, part_id: int) -> Optional[int]:
        """
        Trace /PART -> /PROP -> activation /FUNCT for a single part ID.

        This is the documented chain:
        /PART (part_id) -> property ID in part data -> /PROP/TYPE46 (prop_id) ->
        fct_ID1 field (field 5 of data line 1) -> /FUNCT (funct_id)

        Returns the funct_id, or None if the chain cannot be resolved.
        """
        return self._mapper.part_to_funct.get(part_id)

    # ------------------------------------------------------------------
    # Private: run header injection
    # ------------------------------------------------------------------

    def _inject_run_header(self, parser: DeckParser, config: RunConfig) -> None:
        """
        Insert a comment block at the top of the deck (after #RADIOSS STARTER)
        recording the run parameters.
        """
        header_lines = [
            f"$ --- ParametricEditor run: {config.run_id} ---",
            f"$ t_start={config.t_start:.4f}s  magnitude={config.magnitude_pct:.1f}%MVC"
            f"  baseline={config.baseline_tone_pct:.1f}%MVC  peak_g={config.peak_g:.1f}g",
            f"$ activated_groups: {','.join(config.activated_groups) if config.activated_groups else 'none'}",
        ]
        if config.notes:
            header_lines.append(f"$ notes: {config.notes}")
        header_lines.append("$")

        # Find insertion point: after the first line starting with #RADIOSS or /BEGIN
        insert_after = 0
        for i, line in enumerate(parser._raw_lines):
            upper = line.strip().upper()
            if upper.startswith("#RADIOSS") or upper.startswith("/BEGIN"):
                insert_after = i + 1
                break

        for j, hline in enumerate(header_lines):
            parser._raw_lines.insert(insert_after + j, hline)

        # Re-parse to sync blocks with new lines
        parser._parse()


# ---------------------------------------------------------------------------
# Factory functions for convenience
# ---------------------------------------------------------------------------

def load_editor(
    deck_path: str,
    crash_pulse_funct_id: Optional[int] = None,
    crash_pulse_current_peak_g: Optional[float] = None,
) -> Tuple[ParametricEditor, DeckParser, MuscleMapper]:
    """
    Convenience factory: parse a .rad file and build editor + mapper.

    Parameters
    ----------
    deck_path : str
        Path to the base OpenRadioss .rad deck.
    crash_pulse_funct_id : int, optional
        Explicit crash pulse /FUNCT ID. Auto-detected if None.
    crash_pulse_current_peak_g : float, optional
        Known peak g of the crash pulse. Computed from /FUNCT if None.

    Returns
    -------
    (ParametricEditor, DeckParser, MuscleMapper)
    """
    parser = DeckParser.from_file(deck_path)
    mapper = MuscleMapper(parser)
    editor = ParametricEditor(
        deck_parser=parser,
        muscle_mapper=mapper,
        crash_pulse_funct_id=crash_pulse_funct_id,
        crash_pulse_current_peak_g=crash_pulse_current_peak_g,
    )
    return editor, parser, mapper


# ---------------------------------------------------------------------------
# CLI entry point for batch generation
# ---------------------------------------------------------------------------

def _build_example_batch() -> List[RunConfig]:
    """
    Build an illustrative 3x3x3 factorial design:
    - T_start: -0.020, 0.000, +0.020 s
    - magnitude: 25%, 50%, 100% MVC
    - activated_groups: SCM only, posterior only, all groups
    """
    t_starts = [-0.020, 0.000, 0.020]
    magnitudes = [25.0, 50.0, 100.0]
    group_sets = [
        ["SCM"],
        ["SCap", "SCerv", "CM_C4", "CM_C6"],
        ALL_CERVICAL_GROUPS,
    ]
    group_labels = ["SCM_only", "posterior", "all"]
    configs = []
    for t in t_starts:
        for m in magnitudes:
            for groups, label in zip(group_sets, group_labels):
                run_id = f"T{int(t*1000):+04d}ms_M{int(m):03d}pct_{label}"
                configs.append(
                    RunConfig(
                        run_id=run_id,
                        activated_groups=groups,
                        t_start=t,
                        magnitude_pct=m,
                        baseline_tone_pct=3.5,
                        peak_g=10.0,
                        notes=f"Factorial design: t={t}s, MVC={m}%, groups={label}",
                    )
                )
    return configs


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="OpenRadioss parametric muscle activation editor"
    )
    ap.add_argument("deck", help="Path to base .rad deck file")
    ap.add_argument("output_dir", help="Output directory for batch runs")
    ap.add_argument(
        "--t-start", type=float, default=0.0,
        help="Activation onset time (s), range [-0.06, 0.10]"
    )
    ap.add_argument(
        "--magnitude", type=float, default=50.0,
        help="MVC percent, range [0, 150]"
    )
    ap.add_argument(
        "--baseline", type=float, default=3.5,
        help="Resting tone percent, default 3.5"
    )
    ap.add_argument(
        "--peak-g", type=float, default=10.0,
        help="Crash pulse peak g, range [5, 15]"
    )
    ap.add_argument(
        "--groups", nargs="+", default=ALL_CERVICAL_GROUPS,
        help=f"Activated groups (default: all). Choices: {ALL_CERVICAL_GROUPS}"
    )
    ap.add_argument(
        "--run-id", default="run_001",
        help="Run identifier"
    )
    ap.add_argument(
        "--batch", action="store_true",
        help="Run full example factorial batch instead of single config"
    )
    ap.add_argument(
        "--crash-pulse-funct-id", type=int, default=None,
        help="Explicit crash pulse /FUNCT ID (auto-detected if omitted)"
    )

    args = ap.parse_args()

    if not os.path.exists(args.deck):
        print(f"ERROR: deck file not found: {args.deck}", file=sys.stderr)
        sys.exit(1)

    editor, parser, mapper = load_editor(
        args.deck,
        crash_pulse_funct_id=args.crash_pulse_funct_id,
    )

    print(f"Loaded deck: {args.deck}")
    print(f"Cervical parts found: {len(mapper.cervical_pids)}")
    for g in ALL_CERVICAL_GROUPS:
        pids = mapper.pids_for_group(g)
        functs = mapper.funct_ids_for_group(g)
        print(f"  {g}: {len(pids)} parts, funct_ids={functs}")

    if args.batch:
        configs = _build_example_batch()
        print(f"\nGenerating {len(configs)} batch runs to {args.output_dir} ...")
        paths = editor.generate_batch(configs, args.output_dir)
        for p in paths:
            print(f"  Written: {p}")
        print("Batch complete.")
    else:
        config = RunConfig(
            run_id=args.run_id,
            activated_groups=args.groups,
            t_start=args.t_start,
            magnitude_pct=args.magnitude,
            baseline_tone_pct=args.baseline,
            peak_g=args.peak_g,
        )
        modified = editor.apply_run_config(config)
        paths = editor.generate_batch([config], args.output_dir)
        print(f"Written: {paths[0]}")
