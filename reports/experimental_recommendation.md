# Experimental Recommendation

## Direct Answers
1. Genuinely usable uploaded files: OECD leading indicators top-level CSV, Historic CAPE ratios top-level CSV, china_cli.csv, china_pmi.csv, china_cpi_data.csv, china_industrial_production.csv, China_dividend_yield.csv, China_price_to_earnings.csv, china & em price-to-book ratio.csv, japan_pe_topix_index.csv, em_global_div_yield.csv, em_global_ptb.csv, em_global_pte.csv, BAMLH0A0HYM2 / BAMLH0A0HYM2EY, BAMLHE00EHYIOAS / BAMLHE00EHYIEY.
2. Experimentally worth keeping: The strongest experimental candidate was `baseline_plus_oecd_cape` with `ridge_pooled`, but it still needs to stay experimental until confirmed by a broader benchmark process.
3. China-related additions: China-related additions did not show material, repeatable improvement in this compact rolling test.
4. China recommendation: use only as an experimental macro/valuation block for now; do not add a standalone China sleeve from the uploaded data.
5. Single best experimental extension to carry forward: `baseline_plus_oecd_cape` with `ridge_pooled` as the most defensible next test.
6. Main-design graduation: none of these changes should enter the locked main design yet; they should remain experimental until they beat the frozen baseline more consistently.

## China Sleeve Assessment
- The uploaded SSE Composite workbook is not a USD adjusted-close investable sleeve target, so it does not support a baseline-consistent China sleeve on its own.
- A standalone China sleeve is technically feasible only as a separate ETF-target experiment using the existing swappable target adapter, not from these uploaded files alone.
