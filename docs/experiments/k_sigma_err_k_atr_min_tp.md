# Experiment: k_sigma_err vs k_atr_min_tp (15Min)

## Summary

Grid search over `k_sigma_err` and `k_atr_min_tp` shows a clear optimum around:

- **k_sigma_err = 0.50**
- **k_atr_min_tp = 2.5**

This configuration delivers the highest final equity and total return with a strong Sharpe ratio and acceptable drawdown.

## Raw results (excerpt)

Columns:

- `final_equity`
- `n_trades`
- `total_return`
- `cagr`
- `max_drawdown`
- `sharpe_ratio`
- `win_rate`
- `profit_factor`
- `grid`
- `risk_per_trade_pct`
- `reward_risk_ratio`
- `k_sigma_err`
- `k_atr_min_tp`

Data:

- 40483.777190, 99, 3.048378, 0.623280, -0.161862, 1.910401, 0.393939, 2.114007, noise_filters, 0.02, 3.5, **0.50**, **2.5**
- 37988.052895, 115, 2.798805, 0.587887, -0.173406, 1.761737, 0.365217, 1.907052, noise_filters, 0.02, 3.5, **0.25**, **2.5**
- 40183.099755, 129, 3.018310, 0.619093, -0.161862, 1.757574, 0.356589, 1.704803, noise_filters, 0.02, 3.5, **0.50**, **2.0**
- 36828.490264, 112, 2.682849, 0.570924, -0.190078, 1.741343, 0.366071, 1.858957, noise_filters, 0.02, 3.5, **1.00**, **2.0**
- 35754.413047, 122, 2.575441, 0.554898, -0.173435, 1.666348, 0.352459, 1.703386, noise_filters, 0.02, 3.5, **0.75**, **2.0**

(See source table for full grid.)

## Parameter-wise observations

### k_sigma_err

Tested values: 0.25, 0.50, 0.75, 1.00.

- **0.25**: Decent performance, but generally worse than 0.50 on both equity and Sharpe.
- **0.50**: Consistently appears in the top rows for final equity and Sharpe; drawdowns are not worse than other settings.
- **0.75**: Performance drops relative to 0.50.
- **1.00**: Mixed; some rows are okay, but overall not better than 0.50 and sometimes worse drawdowns.

**Conclusion:** stay around **k_sigma_err ≈ 0.5**. If a more conservative variant is desired, 0.25 is a reasonable backup to test for robustness.

### k_atr_min_tp

Tested values: 1.5, 2.0, 2.5, 3.0, 3.5.

- **1.5**: Often more trades, but lower final equity and Sharpe compared to higher TP multiples.
- **2.0 & 2.5**: Cluster of the best results, including the top configurations. These offer a good balance of returns and drawdown.
- **3.0 & 3.5**: Returns and Sharpe generally deteriorate, even if some drawdowns are similar; the higher TP seems to leave too much on the table.

**Conclusion:** sweet spot is **k_atr_min_tp ∈ [2.0, 2.5]**, with **2.5** slightly ahead in the best row.

## Recommended configuration

If a single configuration must be chosen from this grid:

- **k_sigma_err = 0.5**
- **k_atr_min_tp = 2.5**

If a robustness band is desired for further testing:

- Primary region: `k_sigma_err ∈ {0.5}`, `k_atr_min_tp ∈ {2.0, 2.5}`
- Secondary (for robustness checks): `k_sigma_err ∈ {0.25, 0.75}`, `k_atr_min_tp ∈ {2.0, 2.5}`

This experiment suggests centering future fine-tuning around `k_sigma_err ≈ 0.5` and `k_atr_min_tp ≈ 2.0–2.5`, and treating higher TP multiples (≥3.0) as lower-priority variations.
