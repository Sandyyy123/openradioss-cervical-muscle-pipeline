"""
muscle_mapper.py

Maps neck muscle elements/parts from a CSV descriptor and FE deck parser
into 8 functional groups based on Stemper/Yoganandan SPINE 2006 classification.

Hill-type muscle parameter modification:
- MAT_156 (*MAT_MUSCLE): ALM activation curve + PIS peak isometric stress
- MAT_S15 (*MAT_SPRING_MUSCLE): A activation curve + FMAX peak isometric force

References:
- Stemper & Yoganandan, SPINE 2006 (8-group functional classification)
- Kleinbach et al. PMC5581498 (extended Hill-type LS-DYNA implementation)
- LS-DYNA Keyword Manual Vol II R6.1 (MAT_156, MAT_S15 card definitions)
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Functional muscle group definitions (Stemper/Yoganandan SPINE 2006)
# Keys are group names; values are lists of anatomical substring patterns
# (case-insensitive partial match)
# ---------------------------------------------------------------------------

MUSCLE_GROUPS: dict[str, list[str]] = {
    # ---- FLEXORS ----
    "superficial_flexor": [
        "sternocleidomastoid",
        "scm",
    ],
    "deep_flexor": [
        "longus_capitis",
        "longus_colli",
    ],
    "lateral_flexor": [
        "scalenus",
        "scalene",
    ],
    "anterior_infra": [
        "infrahyoid",
        "suprahyoid",
    ],
    # ---- EXTENSORS ----
    "superficial_extensor": [
        "trapezius",
    ],
    "intermediate_extensor": [
        "splenius_capitis",
    ],
    "deep_extensor": [
        "semispinalis_capitis",
        "semispinalis_cervicis",
        "longissimus_capitis",
        "longissimus_cervicis",
    ],
    "deepest_extensor": [
        "multifidus_cervicis",
    ],
}

# ---------------------------------------------------------------------------
# Dual naming system: anatomical group names <-> parametric_editor.py tokens
# Use GROUP_NAME_MAP to translate between anatomical group names and
# parametric_editor.py abbreviated tokens.
# ---------------------------------------------------------------------------

GROUP_NAME_MAP: dict[str, str] = {
    "superficial_flexor": "SCM",
    "deep_flexor": "CM_C6",
    "lateral_flexor": "Scal",
    "anterior_infra": "STH",
    "superficial_extensor": "Trap",
    "intermediate_extensor": "SCap",
    "deep_extensor": "CM_C4",
    "deepest_extensor": "CM_C6",
}

ABBREVIATED_TO_ANATOMICAL: dict[str, str] = {v: k for k, v in GROUP_NAME_MAP.items()}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class MuscleEntry:
    """Represents one muscle record from the CSV or deck PART."""

    element_id: str
    part_id: str
    name: str               # anatomical name (from CSV 'name' column)
    deck_part_name: str     # PART name string from the FE deck
    side: str = ""          # "left", "right", or "" if not determined
    group: str = ""         # assigned functional group (populated by mapper)
    raw_csv_row: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MuscleMapper
# ---------------------------------------------------------------------------

class MuscleMapper:
    """
    Maps neck muscle elements and parts into 8 functional groups.

    Parameters
    ----------
    csv_path : str
        Path to neck_muscles.csv (passed to NeckMuscleCSVParser or used
        directly if ``deck_parser`` already exposes parsed records).
    deck_parser : object
        Any object that exposes:
          - ``.parts``  - iterable of objects with attributes:
                          ``part_id`` (str), ``name`` (str),
                          ``element_ids`` (list[str])
          - ``.elements`` (optional) - dict mapping element_id -> part_id
        A plain namespace / dict-like object is also accepted; the mapper
        falls back gracefully when attributes are missing.
    """

    # Pattern to detect side from a name string
    _SIDE_RE = re.compile(r"\b(left|right|_l_|_r_|_l\b|_r\b|\.l\.|\.r\.)", re.IGNORECASE)
    _LEFT_RE = re.compile(r"\b(left|_l_|_l\b|\.l\.)", re.IGNORECASE)
    _RIGHT_RE = re.compile(r"\b(right|_r_|_r\b|\.r\.)", re.IGNORECASE)

    def __init__(self, csv_path: str, deck_parser: Any) -> None:
        self.csv_path = csv_path
        self.deck_parser = deck_parser

        # Populated by _load_entries()
        self._entries: list[MuscleEntry] = []
        self._group_map: dict[str, list[MuscleEntry]] | None = None

        # Precompile group patterns for speed
        self._compiled_patterns: dict[str, list[re.Pattern]] = {
            group: [
                re.compile(pat.replace("_", r"[\s_\-]?"), re.IGNORECASE)
                for pat in patterns
            ]
            for group, patterns in MUSCLE_GROUPS.items()
        }

        # Load CSV rows if deck_parser doesn't already hold them
        self._csv_rows: list[dict[str, Any]] = self._load_csv(csv_path)

        # Build the master entry list
        self._load_entries()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_csv(self, csv_path: str) -> list[dict[str, Any]]:
        """Read neck_muscles.csv into a list of row dicts."""
        import csv
        import os

        if not csv_path or not os.path.isfile(csv_path):
            logger.warning("CSV path not found or empty: %r - skipping CSV load", csv_path)
            return []

        rows: list[dict[str, Any]] = []
        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(dict(row))
        logger.debug("Loaded %d rows from %s", len(rows), csv_path)
        return rows

    def _detect_side(self, text: str) -> str:
        """Return 'left', 'right', or '' from a name string."""
        if self._LEFT_RE.search(text):
            return "left"
        if self._RIGHT_RE.search(text):
            return "right"
        return ""

    def _normalise(self, text: str) -> str:
        """Lower-case and collapse separators for matching."""
        return re.sub(r"[\s\-]+", "_", text.lower().strip())

    def _match_group(self, csv_name: str, deck_name: str) -> str:
        """
        Return the first matching functional group name, or "" if none.

        Matching is tried against both the CSV anatomical name and the
        deck PART name string.  The longer / more specific pattern list
        (deepest groups first, then shallower) is checked so that
        "semispinalis_capitis" is caught before a generic "spinalis" would
        be if it ever appeared.
        """
        combined = f"{csv_name} {deck_name}"
        combined_norm = self._normalise(combined)

        # Iterate in insertion order (deepest_extensor last in dict so
        # we check shallower groups first - acceptable because patterns
        # are distinct enough not to conflict).
        for group, compiled in self._compiled_patterns.items():
            for pat in compiled:
                if pat.search(combined_norm):
                    return group
        return ""

    def _load_entries(self) -> None:
        """
        Merge CSV rows with deck parts into a unified entry list.

        Strategy:
        1. Build a map from CSV: element_id / part_id -> (name, row)
        2. Iterate deck_parser.parts; augment with CSV name when available
        3. Any CSV rows not matched to a deck part are added as standalone
           entries (element_id = "", part_id = csv row value or "")
        """
        # --- Step 1: index CSV rows ---
        csv_by_element: dict[str, dict] = {}
        csv_by_part: dict[str, dict] = {}
        csv_consumed: set[int] = set()

        for i, row in enumerate(self._csv_rows):
            eid = str(row.get("element_id", row.get("Element_ID", ""))).strip()
            pid = str(row.get("part_id", row.get("Part_ID", ""))).strip()
            if eid:
                csv_by_element[eid] = row
            if pid:
                csv_by_part[pid] = row
            _ = i  # used later via enumeration if needed

        # --- Step 2: iterate deck parts ---
        parts = getattr(self.deck_parser, "parts", []) or []
        element_to_part: dict[str, str] = {}

        for part in parts:
            part_id = str(getattr(part, "part_id", getattr(part, "id", ""))).strip()
            deck_part_name = str(getattr(part, "name", "")).strip()
            element_ids: list[str] = list(getattr(part, "element_ids", []) or [])

            # Try to get CSV name via part_id lookup first
            csv_row = csv_by_part.get(part_id, {})

            # Fall back: check any element_id belonging to this part
            if not csv_row and element_ids:
                for eid in element_ids:
                    if eid in csv_by_element:
                        csv_row = csv_by_element[eid]
                        break

            csv_name = str(
                csv_row.get("name", csv_row.get("Name", ""))
            ).strip() if csv_row else ""

            side = self._detect_side(csv_name) or self._detect_side(deck_part_name)

            # One entry per element_id (preserves granularity for FE solver)
            if element_ids:
                for eid in element_ids:
                    entry = MuscleEntry(
                        element_id=eid,
                        part_id=part_id,
                        name=csv_name,
                        deck_part_name=deck_part_name,
                        side=side,
                        raw_csv_row=csv_row,
                    )
                    entry.group = self._match_group(csv_name, deck_part_name)
                    self._entries.append(entry)
                    element_to_part[eid] = part_id
            else:
                # Part with no explicit element list
                entry = MuscleEntry(
                    element_id="",
                    part_id=part_id,
                    name=csv_name,
                    deck_part_name=deck_part_name,
                    side=side,
                    raw_csv_row=csv_row,
                )
                entry.group = self._match_group(csv_name, deck_part_name)
                self._entries.append(entry)

            # Mark CSV row as consumed
            if csv_row:
                row_pid = str(csv_row.get("part_id", csv_row.get("Part_ID", ""))).strip()
                if row_pid in csv_by_part:
                    csv_by_part.pop(row_pid, None)

        # --- Step 3: add orphan CSV rows (not matched to any deck part) ---
        for row in self._csv_rows:
            pid = str(row.get("part_id", row.get("Part_ID", ""))).strip()
            eid = str(row.get("element_id", row.get("Element_ID", ""))).strip()
            # Skip if already consumed via deck part iteration
            if pid and any(e.part_id == pid for e in self._entries):
                continue
            if eid and eid in element_to_part:
                continue
            csv_name = str(row.get("name", row.get("Name", ""))).strip()
            side = self._detect_side(csv_name)
            entry = MuscleEntry(
                element_id=eid,
                part_id=pid,
                name=csv_name,
                deck_part_name="",
                side=side,
                raw_csv_row=row,
            )
            entry.group = self._match_group(csv_name, "")
            self._entries.append(entry)

        logger.info("MuscleMapper: %d total muscle entries loaded", len(self._entries))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_group_map(self) -> dict[str, dict[str, list[str]]]:
        """
        Return a dict mapping each of the 8 functional group names to
        a sub-dict with keys ``element_ids`` and ``part_ids``.

        Both left and right sides are included in the same group.

        Returns
        -------
        dict with structure::

            {
                "superficial_flexor": {
                    "element_ids": ["101", "102", ...],
                    "part_ids":    ["10", "11", ...],
                },
                ...
            }
        """
        if self._group_map is not None:
            return self._group_map

        result: dict[str, dict[str, list[str]]] = {
            group: {"element_ids": [], "part_ids": []}
            for group in MUSCLE_GROUPS
        }

        seen_elements: dict[str, set[str]] = {g: set() for g in MUSCLE_GROUPS}
        seen_parts: dict[str, set[str]] = {g: set() for g in MUSCLE_GROUPS}

        for entry in self._entries:
            grp = entry.group
            if not grp or grp not in result:
                continue

            if entry.element_id and entry.element_id not in seen_elements[grp]:
                result[grp]["element_ids"].append(entry.element_id)
                seen_elements[grp].add(entry.element_id)

            if entry.part_id and entry.part_id not in seen_parts[grp]:
                result[grp]["part_ids"].append(entry.part_id)
                seen_parts[grp].add(entry.part_id)

        self._group_map = result
        return result

    def get_non_cervical_part_ids(self) -> list[str]:
        """
        Return all part IDs from the deck that are NOT assigned to any
        cervical muscle group.

        Useful for constructing rigid-body or passive-only part sets
        when applying muscle activation only to cervical musculature.

        Returns
        -------
        list[str]
            Sorted list of part ID strings not belonging to any of the
            8 functional muscle groups.
        """
        group_map = self.build_group_map()

        cervical_part_ids: set[str] = set()
        for data in group_map.values():
            cervical_part_ids.update(data["part_ids"])

        all_parts_in_deck: set[str] = set()
        for part in (getattr(self.deck_parser, "parts", []) or []):
            pid = str(getattr(part, "part_id", getattr(part, "id", ""))).strip()
            if pid:
                all_parts_in_deck.add(pid)

        non_cervical = sorted(all_parts_in_deck - cervical_part_ids)
        logger.debug(
            "Non-cervical parts: %d of %d total deck parts",
            len(non_cervical),
            len(all_parts_in_deck),
        )
        return non_cervical

    def validate(self) -> list[str]:
        """
        Return a list of warning strings for muscle entries that could not
        be assigned to any of the 8 functional groups.

        An empty list means all muscle names matched at least one group.

        Returns
        -------
        list[str]
            Human-readable warning messages, one per unmatched entry.
            Each includes the anatomical name, deck PART name, element_id,
            and part_id to help diagnose missing patterns.
        """
        warnings: list[str] = []
        for entry in self._entries:
            if not entry.group:
                msg = (
                    f"UNMATCHED muscle entry - "
                    f"csv_name={entry.name!r} | "
                    f"deck_part_name={entry.deck_part_name!r} | "
                    f"element_id={entry.element_id!r} | "
                    f"part_id={entry.part_id!r}"
                )
                warnings.append(msg)
                logger.warning(msg)
        return warnings

    def summary(self) -> str:
        """
        Return a multi-line summary string showing element and part counts
        per functional group for verification purposes.

        Also logs the summary at INFO level.

        Returns
        -------
        str
            Formatted summary table.
        """
        group_map = self.build_group_map()
        unmatched = self.validate()

        lines: list[str] = [
            "",
            "=" * 60,
            "  MuscleMapper - Functional Group Summary",
            "  (Stemper/Yoganandan SPINE 2006 classification)",
            "=" * 60,
            f"  {'Group':<28} {'Elements':>10} {'Parts':>8}",
            "-" * 60,
        ]

        total_elements = 0
        total_parts = 0
        for group, data in group_map.items():
            n_el = len(data["element_ids"])
            n_pt = len(data["part_ids"])
            total_elements += n_el
            total_parts += n_pt
            lines.append(f"  {group:<28} {n_el:>10} {n_pt:>8}")

        lines.append("-" * 60)
        lines.append(f"  {'TOTAL':<28} {total_elements:>10} {total_parts:>8}")
        lines.append("")
        lines.append(f"  Unmatched entries : {len(unmatched)}")
        lines.append(
            f"  Non-cervical parts: {len(self.get_non_cervical_part_ids())}"
        )
        lines.append("=" * 60)

        result = "\n".join(lines)
        logger.info(result)
        return result

    def entries_for_group(self, group_name: str) -> list[MuscleEntry]:
        """
        Return all MuscleEntry objects belonging to a given functional group.

        Parameters
        ----------
        group_name : str
            One of the 8 keys in MUSCLE_GROUPS.

        Returns
        -------
        list[MuscleEntry]
            May include both left and right side entries.

        Raises
        ------
        KeyError
            If ``group_name`` is not one of the 8 defined groups.
        """
        if group_name not in MUSCLE_GROUPS:
            raise KeyError(
                f"Unknown group {group_name!r}. "
                f"Valid groups: {list(MUSCLE_GROUPS.keys())}"
            )
        return [e for e in self._entries if e.group == group_name]

    def entries_by_side(
        self, group_name: str, side: str
    ) -> list[MuscleEntry]:
        """
        Filter entries for a group to a single anatomical side.

        Parameters
        ----------
        group_name : str
            One of the 8 functional group names.
        side : str
            ``"left"`` or ``"right"``.

        Returns
        -------
        list[MuscleEntry]
        """
        return [
            e for e in self.entries_for_group(group_name)
            if e.side == side.lower()
        ]

    def all_cervical_element_ids(self) -> list[str]:
        """
        Return a flat deduplicated list of all element IDs assigned to
        any of the 8 cervical muscle groups.

        Useful as an include-set when writing activation curves to a
        modified deck.

        Returns
        -------
        list[str]
            Sorted list of element ID strings.
        """
        group_map = self.build_group_map()
        seen: set[str] = set()
        for data in group_map.values():
            seen.update(data["element_ids"])
        return sorted(seen)

    def all_cervical_part_ids(self) -> list[str]:
        """
        Return a flat deduplicated list of all part IDs assigned to
        any of the 8 cervical muscle groups.

        Returns
        -------
        list[str]
            Sorted list of part ID strings.
        """
        group_map = self.build_group_map()
        seen: set[str] = set()
        for data in group_map.values():
            seen.update(data["part_ids"])
        return sorted(seen)
