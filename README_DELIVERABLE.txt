OpenRadioss Parametric Cervical-Muscle Sweep Pipeline
=====================================================
Open DOCUMENTATION.html in a browser for the full beginner + technical guide
(manuscript-style, with literature references and a trial-run result).

Contents
  run_sweep.py            CLI orchestrator (validate / sweep / report)
  parametric_editor.py    muscle activation editor for OpenRadioss .rad decks
  key_muscle_editor.py    muscle activation editor for VIVA+ LS-DYNA .key decks
  deck_parser.py          .rad block parser/writer
  muscle_mapper.py        muscle -> 8 functional groups
  batch_orchestrator.py   parallel OpenRadioss job control
  postprocessor.py        T01 -> metrics (HIC15, head accel, head-T1 angle, neck loads)
  sweep_config_example.json  reference 8-run sweep
  VIVA_BUILD_SOLVE_HANDOFF.md  how the VIVA+ Yoganandan deck builds+solves natively
  DOCUMENTATION.html      full documentation
  verified_example_output/  verified CSV+HTML + trial activation figure

Requirements: Python 3.9+, requirements.txt, OpenRadioss on PATH (incl. th_to_csv).
