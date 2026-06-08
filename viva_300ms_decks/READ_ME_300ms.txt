300 ms end-time decks for the VIVA+ Yoganandan-2000 rear-sled validation.

The ONLY change from the published catalog deck is the simulation end time:
  *CONTROL_TERMINATION  endtim  600.00  ->  300.00   (units: ms)

To use: in each case folder of the VIVA+ vivaplus-validation catalog
(catalog/Yoganandan-2000-Rear/dyna/<case>/), replace
run_RearImpact_Yoganandan2000.k with the matching file here. Everything else
in the deck is unchanged. This roughly halves the solve time.

Cases: 4.3_50F, 4.3_50M, 6.8_50F, 6.8_50M  (g-level x sex).
