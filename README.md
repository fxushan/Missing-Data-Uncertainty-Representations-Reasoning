# experimental setup

`experiment.py` extends `prob_bf.py` with credal sets and fuzzy clustering.
It reproduces Table 1 and Figure 2 of the midterm report, which we adapted in the report.
`experiment_em.py` adds the EM algorithm of the probability branch.
HighChol is then also missing in the training data.

```bash
pip install numpy pandas scikit-learn matplotlib
python experiment.py        # experiments A and B
python experiment_em.py     # experiment C (EM)
python experiment_compare.py  # experiment D 
```

Outputs in `results/`:

- `table_main.csv`: experiment A, HighChol missing in 10-50% of the test set
- `table_scarcity.csv`: experiment B, 150-5000 records for the HighChol model
- `table_timing.csv`: construction time of each representation
- `table_em.csv`: experiment C, HighChol missing in 10/30/50% of the training set;
  complete cases vs EM vs full data (plus optional E2M rows, set `RHOS = []` to skip)
- experiment D (same splits, refits and masks as experiment A; its shared rows reproduce
  `table_main.csv` exactly): `table_compare.csv` (three extra variants: credal set from the
  5-95% range of the refits, fuzzy clustering with H-hat as a weight, vacuous credal set;
  Brier score for HighChol; whether abstentions are targeted), `table_breakeven.csv`
  (up to which cost measuring HighChol pays off), `table_refits.csv` (number of refits B vs
  result and time), `table_cost.csv` (cost and storage per theory), `worked_example.csv`
  (one record through all four theories)
- `fig_main.png`, `fig_scarcity.png`, `fig_em.png`, `fig_tradeoff.png`

`experiment_em.py` and `experiment_compare.py` import from `experiment.py` and change
nothing in it, so the results of experiments A and B stay the same.
