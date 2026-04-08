# XOPTPOE v4 Allocator Refinement Report

## Scope
- Fixed predictor: `elastic_net__core_plus_interactions__separate_60`.
- Fixed branch: accepted v4 benchmark roster and current walk-forward logic.
- Only allocator settings moved in a narrow second-stage grid between the raw anchor and the over-defensive identity setting.

## Grid
- `lam8_kap0.05_identity`: lambda=8.00, kappa=0.05, omega=identity.
- `lam10_kap0.1_diag`: lambda=10.00, kappa=0.10, omega=diag.
- `lam10_kap0.15_diag`: lambda=10.00, kappa=0.15, omega=diag.
- `lam10_kap0.05_identity`: lambda=10.00, kappa=0.05, omega=identity.
- `lam10_kap0.2_diag`: lambda=10.00, kappa=0.20, omega=diag.
- `lam12_kap0.15_diag`: lambda=12.00, kappa=0.15, omega=diag.
- `lam12_kap0.2_diag`: lambda=12.00, kappa=0.20, omega=diag.
- `lam8_kap0.1_identity`: lambda=8.00, kappa=0.10, omega=identity.
- `lam10_kap0.1_identity`: lambda=10.00, kappa=0.10, omega=identity.
- `lam15_kap0.15_diag`: lambda=15.00, kappa=0.15, omega=diag.
- `lam12_kap0.1_identity`: lambda=12.00, kappa=0.10, omega=identity.
- `lam15_kap0.2_diag`: lambda=15.00, kappa=0.20, omega=diag.
- `lam15_kap0.25_diag`: lambda=15.00, kappa=0.25, omega=diag.
- `lam12_kap0.15_identity`: lambda=12.00, kappa=0.15, omega=identity.
- `lam20_kap0.2_diag`: lambda=20.00, kappa=0.20, omega=diag.

## Candidate Summary
- lam8_kap0.05_identity: end_wealth=2.331, ew_end=1.797, delta=0.534, sharpe=8.164, avg_max_weight=0.455, eff_n=3.44, top_weight=EQ_US.
- lam10_kap0.1_diag: end_wealth=2.300, ew_end=1.797, delta=0.503, sharpe=7.277, avg_max_weight=0.574, eff_n=2.06, top_weight=EQ_US.
- lam10_kap0.15_diag: end_wealth=2.231, ew_end=1.797, delta=0.434, sharpe=7.365, avg_max_weight=0.550, eff_n=2.17, top_weight=EQ_US.
- lam10_kap0.05_identity: end_wealth=2.198, ew_end=1.797, delta=0.401, sharpe=8.253, avg_max_weight=0.427, eff_n=3.59, top_weight=EQ_US.
- lam10_kap0.2_diag: end_wealth=2.167, ew_end=1.797, delta=0.370, sharpe=7.433, avg_max_weight=0.520, eff_n=2.34, top_weight=EQ_US.
- lam12_kap0.15_diag: end_wealth=2.093, ew_end=1.797, delta=0.296, sharpe=7.604, avg_max_weight=0.512, eff_n=2.25, top_weight=EQ_US.
- lam12_kap0.2_diag: end_wealth=2.045, ew_end=1.797, delta=0.248, sharpe=7.602, avg_max_weight=0.488, eff_n=2.41, top_weight=EQ_US.
- lam8_kap0.1_identity: end_wealth=2.036, ew_end=1.797, delta=0.239, sharpe=6.758, avg_max_weight=0.293, eff_n=5.81, top_weight=EQ_US.
- lam10_kap0.1_identity: end_wealth=1.986, ew_end=1.797, delta=0.189, sharpe=6.696, avg_max_weight=0.286, eff_n=5.80, top_weight=EQ_US.
- lam15_kap0.15_diag: end_wealth=1.950, ew_end=1.797, delta=0.153, sharpe=7.790, avg_max_weight=0.494, eff_n=2.33, top_weight=FI_UST.
- lam12_kap0.1_identity: end_wealth=1.938, ew_end=1.797, delta=0.141, sharpe=6.666, avg_max_weight=0.279, eff_n=5.75, top_weight=EQ_US.
- lam15_kap0.2_diag: end_wealth=1.922, ew_end=1.797, delta=0.125, sharpe=7.705, avg_max_weight=0.471, eff_n=2.50, top_weight=FI_UST.
- lam15_kap0.25_diag: end_wealth=1.897, ew_end=1.797, delta=0.100, sharpe=7.702, avg_max_weight=0.446, eff_n=2.69, top_weight=FI_UST.
- lam12_kap0.15_identity: end_wealth=1.849, ew_end=1.797, delta=0.052, sharpe=5.621, avg_max_weight=0.212, eff_n=7.67, top_weight=EQ_US.
- lam20_kap0.2_diag: end_wealth=1.796, ew_end=1.797, delta=-0.001, sharpe=7.944, avg_max_weight=0.459, eff_n=2.67, top_weight=FI_UST.

## Decisions
- Raw best-performance setting: `lam8_kap0.05_identity` with end wealth 2.331 versus equal weight 1.797.
- Best diversification setting: `lam12_kap0.15_identity` with avg max weight 0.212 and effective N 7.67.
- Best balanced carry-forward setting: `lam12_kap0.15_identity` with end wealth delta 0.052, avg max weight 0.212, and effective N 7.67.
- Versus the raw anchor `lam10_kap0.1_diag`, the balanced setting changes avg max weight by -0.362 and effective N by 5.61.

## China Readout
- Under `lam12_kap0.15_identity`, EQ_CN avg_weight=0.0035, max_weight=0.0179, top_weight_frequency=0.0000.

## Balanced Attribution
- EQ_US: total_active_contribution_vs_equal=0.3486, abs_active_share=0.4993.
- EQ_EZ: total_active_contribution_vs_equal=-0.0565, abs_active_share=0.0810.
- EQ_EM: total_active_contribution_vs_equal=-0.0562, abs_active_share=0.0806.
- CR_US_IG: total_active_contribution_vs_equal=0.0425, abs_active_share=0.0609.
- RE_US: total_active_contribution_vs_equal=0.0316, abs_active_share=0.0453.
