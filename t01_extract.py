#!/usr/bin/env python3
"""
t01_extract.py - Streaming extractor for very large OpenRadioss T01 time-history files.

Why this exists
---------------
A benchmark run that writes a time-history record on every solver cycle (TH output
interval = 0.0) produces an enormous T01. The Yoganandan-2000 8g BM run on Rescale
reached ~1,001,002 cycles, so 00_mainT01 is ~100 GB and th_to_csv expands it to a
~340 GB wide CSV (~3.4x the binary). Loading that with pandas is impossible.

This script reads the th_to_csv wide CSV ONE LINE AT A TIME (constant memory),
keeps only what we report, optionally down-samples the rows in time, and writes
small CSVs that fit in an email / Drive:

  <out>_energy_mass.csv  global energy + mass balance, down-sampled time series
                         (internal/kinetic/hourglass/contact energy, mass, added
                         mass %, dt) - the energy/mass diagnosis curve.
  <out>_parts.csv        per-part SUMMARY (one row per part): peak internal energy,
                         peak hourglass energy, eroded energy, and the FIRST time
                         internal energy goes negative. Non-physical negative IE is
                         the signature of the element/contact collapse; this table
                         localises which part fails (e.g. the ~211 ms event).
  <out>_channels.csv     selected node + section channels (head accel node, neck
                         cross-sections), down-sampled, in raw th_to_csv form so the
                         *validated* pipeline (run_sweep.py report / postprocessor.py
                         MetricsExtractor) computes HIC15, peak head accel g, neck
                         Fx/Fz/My, Head-T1 angle from this small file unchanged.

It does NOT recompute any biomechanics metric itself - that stays in the validated
postprocessor so there is a single source of truth for the metric math.

th_to_csv 'time' for these .key decks is in MILLISECONDS (step ~2.997e-4 ms here),
so --downsample-ms is compared directly against column 0.

Usage
-----
  # On Rescale, where 00_mainT01 already lives:
  th_to_csv 00_mainT01                          # -> 00_mainT01.csv (~340 GB scratch)
  python3 t01_extract.py 00_mainT01.csv --out BM \\
          --node 4115307 --node 1000 --downsample-ms 0.1

  # Or let the script run th_to_csv for you (needs th_to_csv on PATH):
  python3 t01_extract.py 00_mainT01 --out BM --node 4115307 --downsample-ms 0.1

The down-sampled outputs are a few MB regardless of the ~100 GB input.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
from pathlib import Path

# Global balance columns written by th_to_csv at the front of every record,
# matched case-insensitively by exact trimmed header text.
GLOBAL_ENERGY_COLS = [
    "INTERNAL ENERGY", "KINETIC ENERGY", "ROTATION ENERGY", "EXTERNAL WORK",
    "SPRING ENERGY", "CONTACT ENERGY", "HOURGLASS ENERGY",
    "ELASTIC CONTACT ENERGY", "FRICTIONAL CONTACT ENERGY", "DAMPING CONTACT ENERGY",
    "PLASTIC WORK", "MASS", "ADDED MASS", "PERCENTAGE ADDED MASS", "TIME STEP",
    "X-MOMENTUM", "Y-MOMENTUM", "Z-MOMENTUM",
]
PART_FIELDS = ("IE", "KE", "HE", "ERODED")  # th_to_csv per-part energy fields


def _norm(h: str) -> str:
    return h.strip().strip('"').strip().upper()


def _is_binary_t01(path: Path) -> bool:
    """A th_to_csv .csv starts with the text 'time'; a binary T01 does not."""
    if path.suffix.lower() == ".csv":
        return False
    with open(path, "rb") as fh:
        head = fh.read(64)
    return (b"\x00" in head) or (not head.lstrip().lower().startswith(b'"time"'))


def _run_th_to_csv(t01: Path) -> Path:
    exe = (shutil.which("th_to_csv") or shutil.which("th_to_csv_linux64_gf")
           or shutil.which("th_to_csv_linux64_gf_sp"))
    if not exe:
        sys.exit("ERROR: th_to_csv not on PATH. `source <OpenRadioss>/env.sh` first, "
                 "or pass an already-converted .csv.")
    print(f"[th_to_csv] converting {t01.name} (large step; needs scratch disk) ...",
          file=sys.stderr)
    subprocess.run([exe, t01.name], cwd=t01.parent, check=True)
    for cand in (t01.parent / (t01.name + ".csv"), t01.with_suffix(".csv")):
        if cand.exists():
            return cand
    sys.exit(f"ERROR: th_to_csv ran but no CSV found beside {t01}")


def _select(header, nodes, sections):
    """Map header -> column indices for the energy block, channels, and per-part fields."""
    e_idx, e_names = [0], ["time"]
    c_idx, c_names = [0], ["time"]
    # parts: title -> {field: column index}
    parts: dict[str, dict[str, int]] = {}

    want_global = {g.upper() for g in GLOBAL_ENERGY_COLS}
    node_ids = {str(n) for n in nodes}
    sec_ids = {str(s) for s in sections}

    for i, raw in enumerate(header):
        if i == 0:
            continue
        h = _norm(raw)
        toks = h.split()
        if h in want_global:
            e_idx.append(i); e_names.append(h.title()); continue
        # per-part energy: "<title...> IE|KE|HE|ERODED" (exclude TH-*/DATABASE channel rows)
        if len(toks) >= 2 and toks[-1] in PART_FIELDS and not h.startswith(("TH-", "DATABASE")):
            title = " ".join(toks[:-1])
            parts.setdefault(title, {})[toks[-1]] = i
            continue
        # selected node channels (positional 'var' components)
        if node_ids and "NODE" in h and any(nid in toks for nid in node_ids):
            c_idx.append(i); c_names.append(raw.strip().strip('"').strip()); continue
        # selected section / interface channels
        if sec_ids and ("SECT" in h or "INTER" in h) and any(sid in toks for sid in sec_ids):
            c_idx.append(i); c_names.append(raw.strip().strip('"').strip()); continue
    return (e_idx, e_names), (c_idx, c_names), parts


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="T01 binary OR already-converted th_to_csv .csv")
    ap.add_argument("--out", default="extract", help="output prefix (default: extract)")
    ap.add_argument("--node", action="append", default=[],
                    help="node id to keep (repeatable): head accel node + sled node")
    ap.add_argument("--section", action="append", default=[],
                    help="cross-section / interface id to keep (repeatable): neck sections")
    ap.add_argument("--downsample-ms", type=float, default=0.0,
                    help="keep ~1 row per this many ms in the time-series outputs "
                         "(0 = every row). Parts summary always scans every row.")
    ap.add_argument("--ie-neg-tol", type=float, default=1e-3,
                    help="flag a part's internal energy as non-physically negative "
                         "only when it drops below -tol * (its peak positive IE). "
                         "Default 1e-3 ignores floating-point noise.")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"ERROR: {src} not found")
    csv_path = _run_th_to_csv(src) if _is_binary_t01(src) else src

    out = args.out
    em_path = Path(f"{out}_energy_mass.csv")
    ch_path = Path(f"{out}_channels.csv")
    pa_path = Path(f"{out}_parts.csv")

    with open(csv_path, "r", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        (e_idx, e_names), (c_idx, c_names), parts = _select(header, args.node, args.section)
        print(f"[select] {len(header)} cols -> energy={len(e_idx)-1} "
              f"channel-cols={len(c_idx)-1} parts={len(parts)}", file=sys.stderr)
        if args.node and len(c_idx) == 1:
            print("[warn] no node columns matched - check ids vs 00_mainT01_TITLES",
                  file=sys.stderr)

        # Per-part running aggregates (constant memory: ~#parts floats).
        # A part is "non-physical" when its internal energy goes negative with
        # magnitude comparable to its own peak positive IE - this ignores
        # floating-point noise (e.g. -1e-30) and only flags a genuine collapse.
        rel_tol = args.ie_neg_tol
        agg = {t: {"ie_min": math.inf, "ie_max": -math.inf, "he_max": -math.inf,
                   "eroded_max": -math.inf, "t_ie_neg": None} for t in parts}

        em_f = open(em_path, "w", newline=""); em_w = csv.writer(em_f); em_w.writerow(e_names)
        ch_f = ch_w = None
        if len(c_idx) > 1:
            ch_f = open(ch_path, "w", newline=""); ch_w = csv.writer(ch_f); ch_w.writerow(c_names)

        last_t, kept, total = None, 0, 0
        for row in reader:
            total += 1
            t = _f(row[0]) if row else None
            # parts: scan EVERY row so we never miss a transient spike
            for title, fields in parts.items():
                a = agg[title]
                if "IE" in fields:
                    v = _f(row[fields["IE"]]) if fields["IE"] < len(row) else None
                    if v is not None:
                        if v < a["ie_min"]: a["ie_min"] = v
                        if v > a["ie_max"]: a["ie_max"] = v
                        # flag only a genuine collapse: negative beyond rel_tol of
                        # the part's own peak positive IE (ignores ~0 noise).
                        if (a["t_ie_neg"] is None and t is not None and a["ie_max"] > 0
                                and v < -rel_tol * a["ie_max"]):
                            a["t_ie_neg"] = t
                if "HE" in fields:
                    v = _f(row[fields["HE"]]) if fields["HE"] < len(row) else None
                    if v is not None and v > a["he_max"]: a["he_max"] = v
                if "ERODED" in fields:
                    v = _f(row[fields["ERODED"]]) if fields["ERODED"] < len(row) else None
                    if v is not None and v > a["eroded_max"]: a["eroded_max"] = v
            # time-series outputs: down-sampled
            if args.downsample_ms > 0 and t is not None and last_t is not None \
                    and (t - last_t) < args.downsample_ms - 1e-12:
                continue
            last_t = t
            em_w.writerow([row[i] if i < len(row) else "" for i in e_idx])
            if ch_w:
                ch_w.writerow([row[i] if i < len(row) else "" for i in c_idx])
            kept += 1

        em_f.close()
        if ch_f:
            ch_f.close()

        # Parts summary, sorted to surface the failure: earliest negative IE first,
        # then largest hourglass energy.
        with open(pa_path, "w", newline="") as pf:
            pw = csv.writer(pf)
            pw.writerow(["part_title", "ie_min", "ie_max", "he_max", "eroded_max",
                         "first_t_ms_ie_negative"])
            def sort_key(item):
                t = item[1]["t_ie_neg"]
                return (t if t is not None else math.inf, -item[1]["he_max"])
            for title, a in sorted(agg.items(), key=sort_key):
                pw.writerow([title,
                             "" if a["ie_min"] == math.inf else f"{a['ie_min']:.6g}",
                             "" if a["ie_max"] == -math.inf else f"{a['ie_max']:.6g}",
                             "" if a["he_max"] == -math.inf else f"{a['he_max']:.6g}",
                             "" if a["eroded_max"] == -math.inf else f"{a['eroded_max']:.6g}",
                             "" if a["t_ie_neg"] is None else f"{a['t_ie_neg']:.4g}"])

    print(f"[done] scanned {total} records, wrote {kept} time-series rows", file=sys.stderr)
    print(f"        {em_path}", file=sys.stderr)
    if ch_f is not None:
        print(f"        {ch_path}", file=sys.stderr)
    print(f"        {pa_path}  ({len(parts)} parts; sorted by earliest negative IE)",
          file=sys.stderr)
    # Surface the headline finding immediately.
    flagged = [(t, a["t_ie_neg"]) for t, a in agg.items() if a["t_ie_neg"] is not None]
    if flagged:
        flagged.sort(key=lambda x: x[1])
        print(f"[diagnosis] {len(flagged)} part(s) reached NEGATIVE internal energy "
              f"(non-physical). Earliest: '{flagged[0][0]}' at t={flagged[0][1]:.4g} ms",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
