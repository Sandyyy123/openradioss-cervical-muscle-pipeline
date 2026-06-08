"""
key_muscle_editor.py - parametric cervical-muscle activation editor for VIVA+ LS-DYNA .key decks.

OpenRadioss reads LS-DYNA .key input directly. The VIVA+ neck muscles are *MAT_MUSCLE
(LS-DYNA material 156) elements whose activation level is driven by a load curve referenced
through the ALM field (a negative LCID on the material's second card). Activation-vs-time is a
*DEFINE_CURVE. This module applies a parametric activation profile (onset time T_start, peak
%MVC, resting baseline tone) to selected cervical muscle groups by:

  1. mapping each muscle *PART title to one of the 8 functional groups,
  2. building a per-run activation *DEFINE_CURVE for the activated groups (ramp to %MVC at
     T_start) and a baseline curve for the rest,
  3. re-pointing each *MAT_MUSCLE ALM field to the appropriate new curve,
  4. writing a modified .key (the rest of the deck is passed through verbatim).

This is the LS-DYNA .key counterpart of parametric_editor.py (which targets OpenRadioss .rad).
Units follow the VIVA+ deck: mm-ms-kg-kN, so time is in milliseconds.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 8 functional cervical groups: token -> anatomical substrings (case-insensitive).
MUSCLE_GROUP_PATTERNS: Dict[str, List[str]] = {
    "SCM":   ["sternocleidomastoid", "scm"],
    "STH":   ["sternohyoid", "sternothyroid", "omohyoid", "thyrohyoid"],
    "Scal":  ["scalenus", "scalene"],
    "Trap":  ["trapezius"],
    "SCap":  ["splenius-capitis", "splenius-cap", "spleniuscapitis"],
    "SCerv": ["splenius-cervicis", "splenius-cerv"],
    "CM_C4": ["semispinalis-capitis", "longissimus-capitis"],
    "CM_C6": ["semispinalis-cervicis", "longissimus-cervicis", "multifidus",
              "longus-capitis", "longus-colli", "iliocostalis", "interspinalis",
              "intertransversarii", "rectus-capitis", "obliquus-capitis", "levator"],
}
ALL_CERVICAL_GROUPS = list(MUSCLE_GROUP_PATTERNS.keys())

_F = 10  # LS-DYNA fixed field width


def _field(line: str, idx: int) -> str:
    return line[idx * _F:(idx + 1) * _F].strip()


def group_of(title: str) -> Optional[str]:
    t = title.lower()
    for token, subs in MUSCLE_GROUP_PATTERNS.items():
        for s in subs:
            if s in t:
                return token
    return None


@dataclass
class MuscleRunConfig:
    run_id: str
    activated_groups: List[str] = field(default_factory=list)
    t_start_s: float = 0.0          # activation onset, seconds (negative = pre-impact)
    magnitude_pct: float = 0.0      # peak activation, % MVC
    baseline_tone_pct: float = 3.5  # resting tone for non-activated groups, % MVC
    rise_ms: float = 5.0            # ramp duration to peak
    end_ms: float = 300.0           # curve end time


class KeyMuscleEditor:
    """Parses a VIVA+ muscle .key and writes parametrically-activated copies."""

    def __init__(self, key_path: str) -> None:
        self.path = Path(key_path)
        if not self.path.exists():
            raise FileNotFoundError(key_path)
        self.lines = self.path.read_text(errors="replace").splitlines()
        self._parts: Dict[str, str] = {}      # mid -> group token (via the part that uses it)
        self._mat_alm_lines: Dict[int, int] = {}   # line index of each *MAT_MUSCLE 2nd data card -> mid
        self._mat_mid_at: Dict[int, str] = {}
        self._max_lcid = 0
        self._parse()

    # ------------------------------------------------------------------
    def _data_lines_after(self, kw_idx: int, n: int, skip_title: bool = False) -> List[int]:
        """Return indices of the next n non-comment data lines after a keyword line.

        ANSA-written blocks (*PART, *..._TITLE) put a free-text name on the first
        non-comment line; skip_title drops it so the real data cards are returned.
        """
        out, i, dropped = [], kw_idx + 1, False
        while i < len(self.lines) and len(out) < n:
            s = self.lines[i].strip()
            if s and not s.startswith("$") and not s.startswith("*"):
                if skip_title and not dropped:
                    dropped = True
                else:
                    out.append(i)
            elif s.startswith("*"):
                break
            i += 1
        return out

    @staticmethod
    def _titled(kw: str) -> bool:
        ku = kw.strip().upper()
        return ku.endswith("_TITLE") or ku.startswith("*PART")

    def _parse(self) -> None:
        # 1. *PART blocks: PID/SECID/MID on the data card, title on the line under the keyword.
        for i, l in enumerate(self.lines):
            ls = l.strip()
            if ls.startswith("*PART"):
                # title is the first non-comment line after the keyword
                title = ""
                j = i + 1
                while j < len(self.lines):
                    sj = self.lines[j].strip()
                    if sj.startswith("$"):
                        j += 1; continue
                    title = sj
                    break
                data = self._data_lines_after(i, 1, skip_title=True)
                if data:
                    # PID SECID MID are the first three fields and always present;
                    # split on whitespace so it works for any LS-DYNA field width.
                    toks = self.lines[data[0]].split()
                    mid = toks[2] if len(toks) >= 3 else _field(self.lines[data[0]], 2)
                    g = group_of(title)
                    if g and mid:
                        self._parts[mid] = g
            elif ls.startswith("*DEFINE_CURVE"):
                data = self._data_lines_after(i, 1, skip_title=self._titled(ls))
                if data:
                    try:
                        self._max_lcid = max(self._max_lcid, int(float(_field(self.lines[data[0]], 0))))
                    except ValueError:
                        pass
            elif ls.startswith("*MAT_MUSCLE"):
                # card1 (MID...) then card2 (ALM SFR SVS SVR SSP). Need the 2 data cards.
                data = self._data_lines_after(i, 2, skip_title=self._titled(ls))
                if len(data) == 2:
                    mid = _field(self.lines[data[0]], 0)
                    self._mat_alm_lines[data[1]] = mid       # line of ALM card
                    self._mat_mid_at[data[1]] = mid

    # ------------------------------------------------------------------
    def group_summary(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for mid, g in self._parts.items():
            out[g] = out.get(g, 0) + 1
        return out

    def _activation_curve(self, lcid: int, peak_frac: float, t_start_ms: float,
                          rise_ms: float, end_ms: float, baseline_frac: float) -> List[str]:
        """Build a *DEFINE_CURVE block (activation vs time, ms) as text lines."""
        t0 = -50.0 if t_start_ms > -50.0 else t_start_ms - 10.0
        pts: List[Tuple[float, float]] = [
            (t0, baseline_frac),
            (t_start_ms, baseline_frac),
            (t_start_ms + rise_ms, peak_frac),
            (end_ms, peak_frac),
        ]
        out = ["*DEFINE_CURVE_TITLE", f"Activation_LCID_{lcid}",
               "$     LCID|     SIDR|      SFA|      SFO|     OFFA|     OFFO|   DATTYP|    LCINT|",
               f"{lcid:>10d}{0:>10d}{1.0:>10.1f}{1.0:>10.1f}{0.0:>10.1f}{0.0:>10.1f}{0:>10d}"]
        for a, o in pts:
            out.append(f"{a:>20.6g}{o:>20.6g}")
        return out

    def apply_run_config(self, cfg: MuscleRunConfig, out_path: str) -> str:
        """Write a parametrically-activated copy of the deck to out_path. Returns out_path."""
        peak_frac = cfg.magnitude_pct / 100.0
        base_frac = cfg.baseline_tone_pct / 100.0
        t_start_ms = cfg.t_start_s * 1000.0

        act_lcid = self._max_lcid + 1
        base_lcid = self._max_lcid + 2
        activated = {g for g in cfg.activated_groups}
        # Guard against silent no-ops: warn if a requested group matches no muscle.
        known = set(self._parts.values())
        for g in activated:
            if g not in known:
                import sys
                print(f"WARNING: activated group '{g}' matched no muscle part "
                      f"(known groups: {sorted(known)}); it will have no effect.",
                      file=sys.stderr)

        new_lines = list(self.lines)
        # 1. re-point each *MAT_MUSCLE ALM to the activation or baseline curve by group
        for ln_idx, mid in self._mat_alm_lines.items():
            g = self._parts.get(mid)
            target = act_lcid if (g in activated) else base_lcid
            orig = new_lines[ln_idx]
            rest = orig[_F:] if len(orig) > _F else ""
            # ALM as negative LCID reference, fixed 10-col field, LS-DYNA float style
            new_lines[ln_idx] = f"{('-' + str(target) + '.'):>10s}" + rest

        # 2. append the two new activation curves before *END
        curves = self._activation_curve(act_lcid, peak_frac, t_start_ms, cfg.rise_ms, cfg.end_ms, base_frac)
        curves += self._activation_curve(base_lcid, base_frac, t_start_ms, cfg.rise_ms, cfg.end_ms, base_frac)
        # insert before a trailing *END if present, else append
        end_idx = next((k for k in range(len(new_lines) - 1, -1, -1)
                        if new_lines[k].strip().upper() == "*END"), None)
        if end_idx is not None:
            new_lines = new_lines[:end_idx] + curves + new_lines[end_idx:]
        else:
            new_lines += curves

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("\n".join(new_lines) + "\n")
        return out_path


def _main() -> int:
    import argparse, json
    ap = argparse.ArgumentParser(description="Parametric VIVA+ .key muscle activation editor")
    ap.add_argument("--key", required=True, help="VIVA+ muscle .key file")
    ap.add_argument("--config", help="JSON with run_id, activated_groups, t_start, magnitude_pct, ...")
    ap.add_argument("--out", help="output .key path")
    ap.add_argument("--summary", action="store_true", help="print group mapping summary and exit")
    a = ap.parse_args()
    ed = KeyMuscleEditor(a.key)
    if a.summary:
        print("Muscle group mapping (group -> #parts):")
        for g, n in sorted(ed.group_summary().items()):
            print(f"  {g:6s} {n}")
        return 0
    d = json.load(open(a.config))
    cfg = MuscleRunConfig(
        run_id=d.get("run_id", "run"),
        activated_groups=d.get("activated_groups", []),
        t_start_s=d.get("t_start", 0.0),
        magnitude_pct=d.get("magnitude_pct", 0.0),
        baseline_tone_pct=d.get("baseline_tone_pct", 3.5),
    )
    out = ed.apply_run_config(cfg, a.out)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
