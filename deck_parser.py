"""
deck_parser.py - OpenRadioss .rad deck file parser for VIVA+ HBM parametric editing.

Supports:
- Parsing keyword blocks from .rad / .k deck files
- /FUNCT, /IMPVEL, /PROP/TYPE46, /ACCEL, /TH/* blocks
- Fixed-width (100-char, 10-char fields) and free-format lines
- NeckMuscleCSVParser for neck_muscles.csv (or auto-generated equivalent)
- Write-back with full formatting preservation
"""

from __future__ import annotations

import os
import re
import csv
import logging
from pathlib import Path
from typing import Any

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Field-level helpers
# ---------------------------------------------------------------------------

FIELD_WIDTH = 10
LINE_WIDTH = 100


def _parse_fixed_fields(line: str) -> list[str]:
    """Split a line into 10-char fields. Returns up to 10 fields."""
    fields: list[str] = []
    padded = line.ljust(LINE_WIDTH)
    for i in range(0, LINE_WIDTH, FIELD_WIDTH):
        fields.append(padded[i:i + FIELD_WIDTH])
    return fields


def _write_fixed_fields(fields: list[str], n_fields: int = 10) -> str:
    """Pack a list of string values into a fixed-width 100-char line."""
    parts: list[str] = []
    for val in fields[:n_fields]:
        parts.append(str(val).rjust(FIELD_WIDTH))
    while len(parts) < n_fields:
        parts.append(" " * FIELD_WIDTH)
    return "".join(parts)


def _is_comment(line: str) -> bool:
    """Return True if line is a comment (starts with # or $) but is not a
    special directive (#RADIOSS STARTER, #enddata, #include)."""
    stripped = line.strip()
    if stripped.startswith("$"):
        return True
    if stripped.startswith("#"):
        upper = stripped.upper()
        if (
            upper.startswith("#RADIOSS")
            or upper.startswith("#ENDDATA")
            or upper.startswith("#INCLUDE")
        ):
            return False
        return True
    return False


def _is_keyword(line: str) -> bool:
    """Return True if line is a keyword header (starts with /)."""
    stripped = line.strip()
    return stripped.startswith("/") and not stripped.startswith("//")


def _keyword_parts(line: str) -> list[str]:
    """Split '/PROP/TYPE46/101/1' into ['PROP', 'TYPE46', '101', '1']."""
    stripped = line.strip().lstrip("/")
    return [p.strip() for p in stripped.split("/") if p.strip() != ""]


# ---------------------------------------------------------------------------
# DeckParser
# ---------------------------------------------------------------------------

class DeckParser:
    """
    Parse and manipulate an OpenRadioss .rad (or .k) deck file.

    Usage
    -----
    parser = DeckParser("model.rad")
    parser.parse()
    functs = parser.get_keyword_blocks("FUNCT")
    f101 = parser.get_funct_by_id("101")
    parser.write("model_modified.rad")
    """

    def __init__(self, deck_path: str) -> None:
        self.deck_path = Path(deck_path)
        if not self.deck_path.exists():
            raise FileNotFoundError(f"Deck file not found: {deck_path}")
        self._raw_lines: list[str] = []
        self._blocks: list[dict[str, Any]] = []
        self._parsed: bool = False

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _read_file(self) -> None:
        with open(self.deck_path, "r", encoding="utf-8", errors="replace") as fh:
            self._raw_lines = fh.readlines()
        # Ensure lines retain their newline; strip only for comparison
        logger.debug("Read %d lines from %s", len(self._raw_lines), self.deck_path)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse(self) -> dict[str, Any]:
        """
        Parse the deck file into structured blocks.

        Returns
        -------
        dict with keys:
            'header_lines': list of str (preamble before first keyword)
            'blocks': list of block dicts
            'footer_lines': list of str (lines after /END)
        """
        if not self._raw_lines:
            self._read_file()

        header_lines: list[str] = []
        footer_lines: list[str] = []
        blocks: list[dict[str, Any]] = []

        in_header = True
        in_footer = False
        current_block: dict[str, Any] | None = None

        for lineno, raw in enumerate(self._raw_lines):
            line = raw.rstrip("\n").rstrip("\r")

            # After /END everything is footer
            if in_footer:
                footer_lines.append(raw)
                continue

            stripped = line.strip().upper()

            if stripped in ("/END", "#ENDDATA"):
                in_footer = True
                if current_block is not None:
                    blocks.append(current_block)
                    current_block = None
                footer_lines.append(raw)
                continue

            if in_header:
                if _is_keyword(line):
                    in_header = False
                    # Start first block
                    current_block = self._new_block(line, lineno)
                else:
                    header_lines.append(raw)
                    continue
            else:
                if _is_keyword(line):
                    # Close previous block
                    if current_block is not None:
                        blocks.append(current_block)
                    current_block = self._new_block(line, lineno)
                else:
                    if current_block is not None:
                        current_block["body_lines"].append(raw)

        # Close last block if file ended without /END
        if current_block is not None:
            blocks.append(current_block)

        self._blocks = blocks
        self._header_lines = header_lines
        self._footer_lines = footer_lines
        self._parsed = True

        # Post-process block internals
        for blk in self._blocks:
            self._post_process_block(blk)

        result = {
            "header_lines": header_lines,
            "blocks": blocks,
            "footer_lines": footer_lines,
        }
        logger.info(
            "Parsed %d blocks from %s", len(blocks), self.deck_path.name
        )
        return result

    def _new_block(self, keyword_line: str, lineno: int) -> dict[str, Any]:
        parts = _keyword_parts(keyword_line)
        keyword = parts[0].upper() if parts else ""
        sub_keywords = parts[1:] if len(parts) > 1 else []
        # Extract numeric ID if present
        block_id: str | None = None
        for part in sub_keywords:
            if re.match(r"^\d+$", part):
                block_id = part
                break
        return {
            "keyword": keyword,
            "sub_keywords": sub_keywords,
            "block_id": block_id,
            "keyword_line": keyword_line,
            "keyword_line_raw": keyword_line + "\n",
            "body_lines": [],
            "lineno": lineno,
            # Parsed fields filled by post_process
            "title": None,
            "data": {},
        }

    def _post_process_block(self, blk: dict[str, Any]) -> None:
        """Extract title and structured data from body_lines."""
        non_comment_lines = [
            ln for ln in blk["body_lines"] if not _is_comment(ln.rstrip("\n"))
        ]
        if non_comment_lines:
            blk["title"] = non_comment_lines[0].rstrip("\n").strip()

        kw = blk["keyword"]
        if kw == "FUNCT" or kw == "FUNCT_SMOOTH":
            self._parse_funct_data(blk, non_comment_lines)
        elif kw == "IMPVEL" or kw == "IMPACC":
            self._parse_impvel_data(blk, non_comment_lines)
        elif kw == "PROP":
            self._parse_prop_data(blk, non_comment_lines)
        elif kw == "ACCEL":
            self._parse_accel_data(blk, non_comment_lines)

    def _parse_funct_data(
        self, blk: dict[str, Any], non_comment: list[str]
    ) -> None:
        """Parse /FUNCT body: scale line + XY pairs."""
        data: dict[str, Any] = {}
        if len(non_comment) < 1:
            blk["data"] = data
            return

        # Line 0: title
        # Line 1: Ascalex  Fscaley  Ashiftx  Fshifty  (optional)
        # Line 2+: X  Y pairs
        xy_start = 1
        if len(non_comment) >= 2:
            scale_line = non_comment[1].rstrip("\n")
            fields = _parse_fixed_fields(scale_line)
            # Try to parse as floats; if first field is non-numeric treat as XY
            try:
                val0 = float(fields[0].strip() or "0")
                data["Ascalex"] = val0
                data["Fscaley"] = float(fields[1].strip() or "1")
                data["Ashiftx"] = float(fields[2].strip() or "0")
                data["Fshifty"] = float(fields[3].strip() or "0")
                data["scale_line_index"] = 1  # index in non_comment
                xy_start = 2
            except ValueError:
                # No scale line present; XY starts at index 1
                xy_start = 1

        xy_pairs: list[tuple[float, float]] = []
        for ln in non_comment[xy_start:]:
            stripped = ln.strip()
            if not stripped:
                continue
            fields = _parse_fixed_fields(ln.rstrip("\n"))
            try:
                x = float(fields[0].strip())
                y = float(fields[1].strip())
                xy_pairs.append((x, y))
            except (ValueError, IndexError):
                continue
        data["xy_pairs"] = xy_pairs
        blk["data"] = data

    def _parse_impvel_data(
        self, blk: dict[str, Any], non_comment: list[str]
    ) -> None:
        """Parse /IMPVEL body: line1=title, line2=IDs, line3=scale."""
        data: dict[str, Any] = {}
        if len(non_comment) >= 2:
            line2 = non_comment[1].rstrip("\n")
            fields = _parse_fixed_fields(line2)
            data["fct_IDT_raw"] = fields[0].strip()
            data["Dir"] = fields[1].strip()
            data["Skew_ID"] = fields[2].strip()
            data["sens_ID"] = fields[3].strip()
            data["grnd_ID"] = fields[4].strip()
            data["frame_ID"] = fields[5].strip()
            data["icoor"] = fields[6].strip()
        if len(non_comment) >= 3:
            line3 = non_comment[2].rstrip("\n")
            fields = _parse_fixed_fields(line3)
            try:
                data["Ascalex"] = float(fields[0].strip() or "1")
            except ValueError:
                data["Ascalex"] = 1.0
            try:
                data["FscaleY"] = float(fields[1].strip() or "1")
            except ValueError:
                data["FscaleY"] = 1.0
            try:
                data["Tstart"] = float(fields[2].strip() or "0")
            except ValueError:
                data["Tstart"] = 0.0
            try:
                data["Tstop"] = float(fields[3].strip() or "1e20")
            except ValueError:
                data["Tstop"] = 1e20
            data["scale_line_index"] = 2
        blk["data"] = data

    def _parse_prop_data(
        self, blk: dict[str, Any], non_comment: list[str]
    ) -> None:
        """Parse /PROP/TYPE46 (SPR_MUSCLE) main data line."""
        sub = [s.upper() for s in blk["sub_keywords"]]
        if "TYPE46" not in sub and "SPR_MUSCLE" not in sub:
            return
        data: dict[str, Any] = {}
        if len(non_comment) >= 2:
            # Title on line 0, data on line 1
            fields = _parse_fixed_fields(non_comment[1].rstrip("\n"))
            try:
                data["Mass"] = float(fields[0].strip() or "0")
                data["Stiffness"] = float(fields[1].strip() or "0")
                data["Vel_max"] = float(fields[2].strip() or "0")
                data["Force"] = float(fields[3].strip() or "0")
                data["Xk"] = float(fields[4].strip() or "0")
                data["fct_ID1"] = fields[5].strip()
                data["fct_ID2"] = fields[6].strip()
                data["fct_ID3"] = fields[7].strip()
                data["fct_ID4"] = fields[8].strip()
                data["Idens"] = fields[9].strip()
            except (ValueError, IndexError):
                pass
        if len(non_comment) >= 3:
            fields2 = _parse_fixed_fields(non_comment[2].rstrip("\n"))
            try:
                data["Damp"] = float(fields2[0].strip() or "0")
                data["EPSI"] = float(fields2[1].strip() or "0")
                data["Scale_t"] = float(fields2[2].strip() or "1")
                data["Scale_x"] = float(fields2[3].strip() or "1")
                data["Scale_v"] = float(fields2[4].strip() or "1")
                data["Scale_F"] = float(fields2[5].strip() or "1")
            except (ValueError, IndexError):
                pass
        blk["data"] = data

    def _parse_accel_data(
        self, blk: dict[str, Any], non_comment: list[str]
    ) -> None:
        """Parse /ACCEL block (accelerometer definition)."""
        data: dict[str, Any] = {}
        if len(non_comment) >= 2:
            fields = _parse_fixed_fields(non_comment[1].rstrip("\n"))
            data["node_ID"] = fields[0].strip()
            data["skew_ID"] = fields[1].strip()
        blk["data"] = data

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def _ensure_parsed(self) -> None:
        if not self._parsed:
            self.parse()

    def get_keyword_blocks(self, keyword: str) -> list[dict[str, Any]]:
        """
        Return all blocks whose top-level keyword matches (case-insensitive).

        Parameters
        ----------
        keyword:
            e.g. 'FUNCT', 'IMPVEL', 'PROP', 'ACCEL', 'TH'

        Returns
        -------
        list of block dicts
        """
        self._ensure_parsed()
        kw_upper = keyword.strip().upper()
        return [b for b in self._blocks if b["keyword"] == kw_upper]

    def get_funct_by_id(self, funct_id: str) -> dict[str, Any] | None:
        """
        Return the /FUNCT (or /FUNCT_SMOOTH) block with the given ID.

        Parameters
        ----------
        funct_id:
            Numeric or string ID, e.g. '101' or 101.

        Returns
        -------
        Block dict, or None if not found.
        """
        self._ensure_parsed()
        fid = str(funct_id).strip()
        for blk in self._blocks:
            if blk["keyword"] in ("FUNCT", "FUNCT_SMOOTH") and blk["block_id"] == fid:
                return blk
        return None

    def get_part_ids_by_name_pattern(self, pattern: str) -> list[str]:
        """
        Regex search across /PART, /SECT, /SECTION_BEAM, /PROP block titles
        and keyword lines.

        Parameters
        ----------
        pattern:
            Python regex pattern, e.g. r'SCM', r'Semispinalis', r'Muscle.*L$'

        Returns
        -------
        list of block_ids (strings) for matching blocks
        """
        self._ensure_parsed()
        compiled = re.compile(pattern, re.IGNORECASE)
        result: list[str] = []
        for blk in self._blocks:
            kw = blk["keyword"]
            if kw not in ("PART", "SECT", "PROP", "SPRING", "MAT"):
                # Also check TH sub-keywords and beam section keywords
                if not any(
                    s in blk["keyword_line"].upper()
                    for s in ("PART", "SECT", "PROP", "SPRING", "BEAM")
                ):
                    continue
            # Search title
            title = blk.get("title") or ""
            kw_line = blk.get("keyword_line") or ""
            if compiled.search(title) or compiled.search(kw_line):
                if blk["block_id"] is not None:
                    result.append(blk["block_id"])
                else:
                    # Try to extract an ID from keyword_line
                    m = re.search(r"/(\d+)", kw_line)
                    if m:
                        result.append(m.group(1))
        return result

    def get_accelero_blocks(self) -> list[dict[str, Any]]:
        """
        Return crash pulse definitions: /IMPVEL, /IMPACC, and /ACCEL blocks.

        Returns
        -------
        list of block dicts
        """
        self._ensure_parsed()
        crash_keywords = {"IMPVEL", "IMPACC", "ACCEL"}
        return [b for b in self._blocks if b["keyword"] in crash_keywords]

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def set_funct_fscaley(self, funct_id: str, new_scale: float) -> bool:
        """
        Update the Fscaley field of a /FUNCT block (crash pulse amplitude).

        Parameters
        ----------
        funct_id:
            ID of the /FUNCT block to update.
        new_scale:
            New Fscaley value.

        Returns
        -------
        True if updated, False if block not found.
        """
        blk = self.get_funct_by_id(funct_id)
        if blk is None:
            logger.warning("FUNCT ID %s not found", funct_id)
            return False
        data = blk.get("data", {})
        scale_idx = data.get("scale_line_index")
        if scale_idx is None:
            logger.warning("FUNCT %s has no scale line", funct_id)
            return False

        # Find the scale line in body_lines (skipping comments)
        nc_idx = 0
        for i, raw in enumerate(blk["body_lines"]):
            if _is_comment(raw.rstrip("\n")):
                continue
            if nc_idx == scale_idx:
                fields = _parse_fixed_fields(raw.rstrip("\n"))
                fields[1] = f"{new_scale:>10.6g}"
                new_line = "".join(fields).rstrip() + "\n"
                blk["body_lines"][i] = new_line
                data["Fscaley"] = new_scale
                logger.info("Set FUNCT %s Fscaley -> %g", funct_id, new_scale)
                return True
            nc_idx += 1
        return False

    def set_funct_ashiftx(self, funct_id: str, new_shift: float) -> bool:
        """
        Update the Ashiftx field of a /FUNCT block (activation onset shift).

        Parameters
        ----------
        funct_id:
            ID of the /FUNCT block to update.
        new_shift:
            New Ashiftx value.
        """
        blk = self.get_funct_by_id(funct_id)
        if blk is None:
            logger.warning("FUNCT ID %s not found", funct_id)
            return False
        data = blk.get("data", {})
        scale_idx = data.get("scale_line_index")
        if scale_idx is None:
            return False

        nc_idx = 0
        for i, raw in enumerate(blk["body_lines"]):
            if _is_comment(raw.rstrip("\n")):
                continue
            if nc_idx == scale_idx:
                fields = _parse_fixed_fields(raw.rstrip("\n"))
                fields[2] = f"{new_shift:>10.6g}"
                new_line = "".join(fields).rstrip() + "\n"
                blk["body_lines"][i] = new_line
                data["Ashiftx"] = new_shift
                logger.info("Set FUNCT %s Ashiftx -> %g", funct_id, new_shift)
                return True
            nc_idx += 1
        return False

    def set_impvel_fscaley(self, impvel_id: str, new_scale: float) -> bool:
        """
        Update the FscaleY field of an /IMPVEL block.

        Parameters
        ----------
        impvel_id:
            ID of the /IMPVEL block.
        new_scale:
            New FscaleY value.
        """
        blk: dict[str, Any] | None = None
        for b in self._blocks:
            if b["keyword"] in ("IMPVEL", "IMPACC") and b["block_id"] == str(impvel_id):
                blk = b
                break
        if blk is None:
            logger.warning("IMPVEL/IMPACC ID %s not found", impvel_id)
            return False
        data = blk.get("data", {})
        scale_idx = data.get("scale_line_index")
        if scale_idx is None:
            return False

        nc_idx = 0
        for i, raw in enumerate(blk["body_lines"]):
            if _is_comment(raw.rstrip("\n")):
                continue
            if nc_idx == scale_idx:
                fields = _parse_fixed_fields(raw.rstrip("\n"))
                fields[1] = f"{new_scale:>10.6g}"
                new_line = "".join(fields).rstrip() + "\n"
                blk["body_lines"][i] = new_line
                data["FscaleY"] = new_scale
                logger.info("Set IMPVEL %s FscaleY -> %g", impvel_id, new_scale)
                return True
            nc_idx += 1
        return False

    def set_muscle_force(self, prop_id: str, new_force: float) -> bool:
        """
        Update the Force (Fmax/MVC) field of a /PROP/TYPE46 block.

        Parameters
        ----------
        prop_id:
            Property ID of the muscle spring.
        new_force:
            New Force value.
        """
        blk: dict[str, Any] | None = None
        for b in self._blocks:
            sub_upper = [s.upper() for s in b.get("sub_keywords", [])]
            if b["keyword"] == "PROP" and b["block_id"] == str(prop_id):
                if "TYPE46" in sub_upper or "SPR_MUSCLE" in sub_upper:
                    blk = b
                    break
        if blk is None:
            logger.warning("PROP/TYPE46 ID %s not found", prop_id)
            return False

        nc_idx = 0
        for i, raw in enumerate(blk["body_lines"]):
            if _is_comment(raw.rstrip("\n")):
                continue
            if nc_idx == 1:  # data line (0 = title)
                fields = _parse_fixed_fields(raw.rstrip("\n"))
                fields[3] = f"{new_force:>10.6g}"
                new_line = "".join(fields).rstrip() + "\n"
                blk["body_lines"][i] = new_line
                blk["data"]["Force"] = new_force
                logger.info("Set PROP %s Force -> %g", prop_id, new_force)
                return True
            nc_idx += 1
        return False

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, output_path: str) -> None:
        """
        Write the (possibly modified) deck back to a file, preserving all
        formatting.

        Parameters
        ----------
        output_path:
            Destination file path. May be the same as input (in-place).
        """
        self._ensure_parsed()
        out = Path(output_path)
        with open(out, "w", encoding="utf-8") as fh:
            # Header
            for line in self._header_lines:
                fh.write(line if line.endswith("\n") else line + "\n")
            # Blocks
            for blk in self._blocks:
                kw_raw = blk["keyword_line_raw"]
                fh.write(kw_raw if kw_raw.endswith("\n") else kw_raw + "\n")
                for body_line in blk["body_lines"]:
                    fh.write(
                        body_line if body_line.endswith("\n") else body_line + "\n"
                    )
            # Footer (/END and beyond)
            for line in self._footer_lines:
                fh.write(line if line.endswith("\n") else line + "\n")
        logger.info("Wrote deck to %s", out)

    # ------------------------------------------------------------------
    # Convenience summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a high-level summary of the parsed deck."""
        self._ensure_parsed()
        kw_counts: dict[str, int] = {}
        for blk in self._blocks:
            kw_counts[blk["keyword"]] = kw_counts.get(blk["keyword"], 0) + 1
        return {
            "file": str(self.deck_path),
            "total_lines": len(self._raw_lines),
            "total_blocks": len(self._blocks),
            "keyword_counts": kw_counts,
        }


# ---------------------------------------------------------------------------
# NeckMuscleCSVParser
# ---------------------------------------------------------------------------

class NeckMuscleCSVParser:
    """
    Parse neck_muscles.csv (or equivalent auto-generated CSV).

    Handles common column name variants:
    - Element_ID / element_id / pid / PID
    - name / muscle_name / PartName / part_name
    - side / Side / laterality
    - pcsa_mm2 / PCSA / area
    - muscle_family / group / Group / MuscleFamily

    If pandas is not installed, falls back to a plain dict-list result.
    """

    # Canonical name -> accepted aliases (lower-case for matching)
    _COLUMN_MAP: dict[str, list[str]] = {
        "element_id": ["element_id", "pid", "part_id", "id", "eid"],
        "name": [
            "name",
            "muscle_name",
            "partname",
            "part_name",
            "title",
            "label",
        ],
        "side": ["side", "laterality", "lat", "hemisphere"],
        "pcsa_mm2": ["pcsa_mm2", "pcsa", "area", "a_mm2", "csa"],
        "muscle_family": [
            "muscle_family",
            "group",
            "family",
            "musclefamily",
            "muscle_group",
            "anatomical_group",
        ],
        "mid": ["mid", "mat_id", "material_id"],
        "secid": ["secid", "sec_id", "section_id"],
    }

    def __init__(self, csv_path: str) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

    def _resolve_columns(
        self, header: list[str]
    ) -> dict[str, str]:
        """
        Map canonical names to actual header names found in file.

        Returns
        -------
        dict: canonical_name -> actual_column_name (only for found columns)
        """
        header_lower = {h.lower().strip(): h for h in header}
        resolved: dict[str, str] = {}
        for canonical, aliases in self._COLUMN_MAP.items():
            for alias in aliases:
                if alias in header_lower:
                    resolved[canonical] = header_lower[alias]
                    break
        return resolved

    def _infer_side_from_name(self, name: str) -> str:
        """Guess side from muscle name string."""
        if not name:
            return ""
        n_upper = name.upper()
        if n_upper.endswith("-L") or "-L-" in n_upper or "_L_" in n_upper:
            return "L"
        if n_upper.endswith("-R") or "-R-" in n_upper or "_R_" in n_upper:
            return "R"
        if n_upper.endswith("-C") or n_upper.endswith("-M"):
            return "C"
        return ""

    def parse(self) -> "pd.DataFrame | list[dict[str, Any]]":
        """
        Parse the CSV and return a DataFrame (if pandas is available)
        or a list of dicts.

        Canonical columns always present in output (filled with '' if missing):
            Element_ID, name, side, pcsa_mm2, muscle_family, mid, secid

        Additional columns from the file are preserved as-is.
        """
        rows: list[dict[str, Any]] = []
        with open(self.csv_path, "r", encoding="utf-8-sig", errors="replace") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                logger.warning("CSV %s appears empty", self.csv_path)
                if _PANDAS_AVAILABLE:
                    return pd.DataFrame()
                return []

            col_map = self._resolve_columns(list(reader.fieldnames))

            for raw_row in reader:
                row: dict[str, Any] = {}

                # Canonical columns
                for canonical in [
                    "element_id",
                    "name",
                    "side",
                    "pcsa_mm2",
                    "muscle_family",
                    "mid",
                    "secid",
                ]:
                    actual = col_map.get(canonical)
                    row[canonical] = raw_row[actual].strip() if actual else ""

                # Preserve all other columns verbatim (case-insensitive dedup)
                row_keys_lower = {k.lower() for k in row}
                for field in reader.fieldnames:
                    if field not in row and field.lower() not in row_keys_lower:
                        row[field] = raw_row[field]

                # Infer side from name if side column missing
                if not row.get("side") and row.get("name"):
                    row["side"] = self._infer_side_from_name(row["name"])

                rows.append(row)

        if not _PANDAS_AVAILABLE:
            logger.warning("pandas not available - returning list of dicts")
            return rows

        df = pd.DataFrame(rows)

        # Rename canonical columns to Title_Case for consistency
        rename_map: dict[str, str] = {
            "element_id": "Element_ID",
            "name": "name",
            "side": "side",
            "pcsa_mm2": "pcsa_mm2",
            "muscle_family": "muscle_family",
            "mid": "mid",
            "secid": "secid",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Type coercions
        for col in ["Element_ID", "mid", "secid"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        if "pcsa_mm2" in df.columns:
            df["pcsa_mm2"] = pd.to_numeric(df["pcsa_mm2"], errors="coerce")

        return df

    @staticmethod
    def from_deck_parser(dp: DeckParser) -> "pd.DataFrame | list[dict[str, Any]]":
        """
        Generate a muscle CSV equivalent directly from a parsed DeckParser.

        Scans /PROP/TYPE46 and /SPRING blocks for muscle elements by matching
        the VIVA+ naming convention (NE-Muscle-* or Muscle in title).

        Returns
        -------
        DataFrame or list of dicts with columns:
            Element_ID, name, side, pcsa_mm2, muscle_family, mid, secid, Force
        """
        dp._ensure_parsed()
        rows: list[dict[str, Any]] = []
        for blk in dp._blocks:
            sub_upper = [s.upper() for s in blk.get("sub_keywords", [])]
            kw = blk["keyword"]
            is_muscle_prop = kw == "PROP" and (
                "TYPE46" in sub_upper or "SPR_MUSCLE" in sub_upper
            )
            # Also pick up PART / SECTION_BEAM that mention "Muscle" in title
            title = blk.get("title") or ""
            is_muscle_named = re.search(r"muscle", title, re.IGNORECASE) is not None

            if not (is_muscle_prop or is_muscle_named):
                continue

            pid = blk.get("block_id") or ""
            data = blk.get("data") or {}
            name = title.strip()
            side = ""
            muscle_family = ""

            # Parse VIVA+ naming: NE-Muscle-SCM-S-Clavicle-1-L
            m = re.match(
                r"(?:NE-)?Muscle-([A-Za-z\-]+?)-(?:[A-Za-z0-9\-]+-)?([LRC])$",
                name,
                re.IGNORECASE,
            )
            if m:
                muscle_family = m.group(1)
                side = m.group(2).upper()
            else:
                # Fallback: last character
                if name.endswith("-L"):
                    side = "L"
                elif name.endswith("-R"):
                    side = "R"

            rows.append(
                {
                    "Element_ID": int(pid) if pid.isdigit() else pid,
                    "name": name,
                    "side": side,
                    "pcsa_mm2": None,  # Not available from PROP/TYPE46 directly
                    "muscle_family": muscle_family,
                    "mid": data.get("fct_ID1", ""),  # placeholder
                    "secid": pid,
                    "Force": data.get("Force"),
                }
            )

        if _PANDAS_AVAILABLE:
            return pd.DataFrame(rows)
        return rows


# ---------------------------------------------------------------------------
# CLI entry point (debug helper)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python deck_parser.py <deck.rad> [output.rad]")
        sys.exit(1)

    dp = DeckParser(sys.argv[1])
    result = dp.parse()
    summary = dp.summary()
    print(json.dumps(summary, indent=2))

    if len(sys.argv) >= 3:
        dp.write(sys.argv[2])
        print(f"Written to {sys.argv[2]}")
