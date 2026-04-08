# XOPTPOE v4 Allocator Sweep Report

## Scope
- Fixed object: `best_60_predictor = elastic_net__core_plus_interactions__separate_60`.
- Fixed branch: v4 accepted benchmark roster and current split logic.
- Only allocator regularization moved: `lambda_risk`, `kappa`, and existing `omega_type`.
- This is a presentation-benchmark tuning pass, not a new predictor selection pass.

## Grid
- `lam30_kap0.5_diag`: lambda=30.00, kappa=0.50, omega=diag.
- `lam20_kap0.5_diag`: lambda=20.00, kappa=0.50, omega=diag.
- `lam30_kap0.1_diag`: lambda=30.00, kappa=0.10, omega=diag.
- `lam30_kap0.25_diag`: lambda=30.00, kappa=0.25, omega=diag.
- `lam15_kap0.5_diag`: lambda=15.00, kappa=0.50, omega=diag.
- `lam20_kap0.1_diag`: lambda=20.00, kappa=0.10, omega=diag.
- `lam20_kap0.25_diag`: lambda=20.00, kappa=0.25, omega=diag.
- `lam15_kap0.1_diag`: lambda=15.00, kappa=0.10, omega=diag.
- `lam15_kap0.25_diag`: lambda=15.00, kappa=0.25, omega=diag.
- `lam10_kap0.25_diag`: lambda=10.00, kappa=0.25, omega=diag.
- `lam10_kap0.1_diag`: lambda=10.00, kappa=0.10, omega=diag.
- `lam15_kap0.25_identity`: lambda=15.00, kappa=0.25, omega=identity.
- `lam20_kap0.5_identity`: lambda=20.00, kappa=0.50, omega=identity.

## Test Results
- lam30_kap0.5_diag: sharpe=8.7643, avg_return=0.0380, avg_max_weight=0.3838, eff_n=3.53, top_weight=FI_UST, top_weight_freq=1.0000, top_active_share=0.3889.
- lam20_kap0.5_diag: sharpe=8.2582, avg_return=0.0416, avg_max_weight=0.3655, eff_n=3.43, top_weight=FI_UST, top_weight_freq=0.7917, top_active_share=0.4372.
- lam30_kap0.1_diag: sharpe=8.2579, avg_return=0.0438, avg_max_weight=0.5123, eff_n=2.53, top_weight=FI_UST, top_weight_freq=1.0000, top_active_share=0.5106.
- lam30_kap0.25_diag: sharpe=8.1467, avg_return=0.0418, avg_max_weight=0.4464, eff_n=2.98, top_weight=FI_UST, top_weight_freq=1.0000, top_active_share=0.4628.
- lam15_kap0.5_diag: sharpe=8.0594, avg_return=0.0443, avg_max_weight=0.3608, eff_n=3.45, top_weight=FI_UST, top_weight_freq=0.5833, top_active_share=0.4742.
- lam20_kap0.1_diag: sharpe=7.9949, avg_return=0.0497, avg_max_weight=0.5122, eff_n=2.29, top_weight=FI_UST, top_weight_freq=0.8750, top_active_share=0.5762.
- lam20_kap0.25_diag: sharpe=7.9470, avg_return=0.0466, avg_max_weight=0.4372, eff_n=2.84, top_weight=FI_UST, top_weight_freq=0.7917, top_active_share=0.5175.
- lam15_kap0.1_diag: sharpe=7.8336, avg_return=0.0548, avg_max_weight=0.5144, eff_n=2.18, top_weight=FI_UST, top_weight_freq=0.5417, top_active_share=0.6182.
- lam15_kap0.25_diag: sharpe=7.7022, avg_return=0.0509, avg_max_weight=0.4458, eff_n=2.69, top_weight=FI_UST, top_weight_freq=0.5417, top_active_share=0.5616.
- lam10_kap0.25_diag: sharpe=7.4362, avg_return=0.0583, avg_max_weight=0.4893, eff_n=2.53, top_weight=EQ_US, top_weight_freq=0.8333, top_active_share=0.6356.
- lam10_kap0.1_diag: sharpe=7.2771, avg_return=0.0645, avg_max_weight=0.5742, eff_n=2.06, top_weight=EQ_US, top_weight_freq=0.9167, top_active_share=0.6762.
- lam15_kap0.25_identity: sharpe=4.0502, avg_return=0.0363, avg_max_weight=0.1540, eff_n=9.80, top_weight=EQ_US, top_weight_freq=0.8750, top_active_share=0.4449.
- lam20_kap0.5_identity: sharpe=3.1222, avg_return=0.0325, avg_max_weight=0.1223, eff_n=11.78, top_weight=FI_UST, top_weight_freq=0.8750, top_active_share=0.3777.

## Raw Best Sharpe
- `lam30_kap0.5_diag` is the raw test-Sharpe winner: sharpe=8.7643, avg_return=0.0380, avg_max_weight=0.3838, eff_n=3.53.

## Balanced Diversification Candidate
- `lam15_kap0.25_identity` is the best balanced setting in this compact sweep: sharpe=4.0502, avg_return=0.0363, avg_max_weight=0.1540, eff_n=9.80, top_active_share=0.4449.
- Versus the current raw anchor `lam10_kap0.1_diag`: return delta=-0.0282, sharpe delta=-3.2268, max-weight delta=-0.4202, eff_n delta=7.74.
- Versus the current heuristic diversified object `best_60_diversified_cap`: sharpe delta=1.0324, max-weight delta=-0.1380, eff_n delta=5.27.

## China Readout
- Under `lam15_kap0.25_identity`, EQ_CN avg_weight=0.0180, max_weight=0.0295, nonzero_alloc_share=1.0000, top_weight_frequency=0.0000.

## Balanced Attribution
- EQ_US: total_active_contribution_vs_equal=0.2019, abs_active_share=0.4449.
- EQ_EZ: total_active_contribution_vs_equal=-0.0504, abs_active_share=0.1110.
- EQ_EM: total_active_contribution_vs_equal=-0.0476, abs_active_share=0.1048.
- CR_US_IG: total_active_contribution_vs_equal=0.0331, abs_active_share=0.0730.
- CR_US_HY: total_active_contribution_vs_equal=0.0232, abs_active_share=0.0512.

## Recommendation
- Carry-forward presentation benchmark: `lam15_kap0.25_identity`. It materially lowers sleeve dominance without collapsing return quality, and it is stronger than the current heuristic diversified object on both Sharpe and breadth.
- Keep `lam30_kap0.5_diag` only as the raw Sharpe reference. It is stronger on pure Sharpe but still more concentration-driven than the recommended balanced setting.
