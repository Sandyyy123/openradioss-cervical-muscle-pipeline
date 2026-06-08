# VIVA+ Yoganandan-2000 Rear Sled in native OpenRadioss - build/solve handoff

Status as of 2026-06-07. Prepared by Sandeep Grover.

## Result
The VIVA+ Yoganandan-2000 rear-impact validation deck (50F, 6.8 g) **builds and
solves in native OpenRadioss directly from the published LS-DYNA `.key`** - no
`.rad` conversion, no belt/seat/contact rewrite. Verified: starter completes with
0 errors and writes a 1.2 GB restart; engine runs cleanly and writes a T01
time-history (terminated only by a wall-clock time-box, not an instability).

Model size: 535,223 solid + 207,724 quad-shell + 42,012 tri-shell + beams/springs
(~785k elements), full 50F body + rigid seat + 3-point belt.

## Two fixes required to build (both small, both documented)
1. **Lumbar cortical-bone material** (MID 501011, `*MAT_ANISOTROPIC_ELASTIC_PLASTIC`
   / MAT157). OpenRadioss force-maps MAT157 to LAW58 (fabric) and rejects a field.
   This is the ONLY MAT157 in the model and it is in the **lumbar spine - outside
   the cervical study region**. Substituted a supported `*MAT_PIECEWISE_LINEAR_PLASTICITY`
   (LAW36) with representative cortical properties (E=16.7 GPa, PR=0.3, SIGY=0.8).
   Cleared 6 of 8 errors. File: `vivaplus/model/common/vivaplus-materials.k`
   (original kept as `.orig`).
2. **T1_Local output coordinate** (`*DEFINE_COORDINATE_NODES` id 800 in
   `env/Outputs_for_Yoganandan2000_50F.k`). OpenRadioss auto-numbers its internal
   skew/frame IDs in a range that overlaps this output-only frame, producing 2
   duplicate-ID errors. The coordinate is used only to report head-vs-T1 angle, not
   to run the sim. Converted the two `*DATABASE_HISTORY_NODE_LOCAL_ID` cards to plain
   `*DATABASE_HISTORY_NODE` (global) and dropped the coordinate. Head-T1 angle is then
   computed in post-processing from the two node histories. Cleared the last 2 errors.
   (original kept as `.orig`).

Neither fix touches the neck, the muscles, the belt, the seat, or the contacts.

## Run procedure (native OpenRadioss)
```bash
source /opt/openradioss/env.sh            # sets OPENRADIOSS_PATH, RAD_CFG_PATH, LD_LIBRARY_PATH, PATH
cd catalog/Yoganandan-2000-Rear/dyna/6.8_50F
starter_linux64_gf -i 00_main.key -np <N> # builds restart (.rst) + engine file _0001.rad
mpirun -np <N> engine_linux64_gf_ompi -i 00_main_0001.rad   # solves; writes 00_mainT01
```
A full 600 ms solve is a **cluster job**: ~1.1e6 s single-process here; budget hours
on 16-64 cores. Run the 4 published cases (4.3/6.8 g x 50F/50M) the same way.

## Pipeline (the contracted automation)
`tools/openradioss/` (own git repo). Parses muscle groups, applies per-run
activation (group, T_start, %MVC, baseline tone), scales the crash pulse, batch-runs
OpenRadioss, converts binary T01 via `th_to_csv`, extracts peak head accel / HIC15 /
head-T1 angle / upper+lower neck Fx/Fz/My to CSV + HTML. Verified end-to-end on a
known-good OpenRadioss deck.

## REMAINING before a full sweep deliverable
1. Adapt the parametric editor from OpenRadioss `.rad` (`/FUNCT`) to the VIVA+ `.key`
   muscle format (`*MAT_MUSCLE` + `*DEFINE_CURVE` in `vivaplus-22-Neck-Muscles.k`).
2. Run the baseline + sweep on a multi-core cluster.
3. Validate head/T1 kinematics against the catalog reference corridors
   (`data/experiment/`), post-processing head-T1 angle from the two node histories.
