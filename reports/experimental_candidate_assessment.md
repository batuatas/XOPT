# Experimental Candidate Assessment

## Used Experimentally
- OECD leading indicators top-level CSV [oecd_leading]: Used for US/Japan CLI and EA20 business-confidence proxy in cross-country bundle.
- Historic CAPE ratios top-level CSV [valuation]: Used for US/Europe/Japan local CAPE and China CAPE in valuation bundles.
- china_cli.csv [china_macro]: Clean monthly China leading indicator; conservatively lagged one month for EM-targeted features.
- china_pmi.csv [china_macro]: Monthly China PMI usable with one-month lag; NBS manufacturing chosen as compact activity proxy.
- china_cpi_data.csv [china_macro]: Monthly headline CPI usable with one-month lag; subcomponents not adopted.
- china_industrial_production.csv [china_macro]: Monthly industrial-production growth usable with one-month lag.
- China_dividend_yield.csv [china_valuation]: Used as China-vs-EM dividend-yield relative valuation signal for EQ_EM.
- China_price_to_earnings.csv [china_valuation]: Used as China-vs-EM PE relative valuation signal for EQ_EM.
- china & em price-to-book ratio.csv [china_valuation]: Used as China-vs-EM PTB relative valuation signal for EQ_EM.
- japan_pe_topix_index.csv [japan_enrichment]: Used as Japan local valuation enrichment.
- em_global_div_yield.csv [global_valuation]: Used for EM-vs-global dividend-yield spread.
- em_global_ptb.csv [global_valuation]: Used for EM-vs-global price-to-book spread.
- em_global_pte.csv [global_valuation]: Used for EM-vs-global price-to-earnings spread.
- BAMLH0A0HYM2 / BAMLH0A0HYM2EY [credit_stress]: Used as additional US HY spread/yield stress features.
- BAMLHE00EHYIOAS / BAMLHE00EHYIEY [credit_stress]: Used as additional European HY spread/yield stress features.

## Usable But Not Used
- china_gdp_cmi.csv [china_macro]: Quarterly/sparse timing makes conservative alignment possible but fragile; omitted from compact bundles.
- japan_macro_fiscal_socioeconomic.csv [japan_enrichment]: Contains mixed-frequency overlaps with baseline and timing-ambiguous series; not clean enough for disciplined first pass.
- JaPaN GDP.csv [japan_enrichment]: Quarterly GDP requires extra conservative lag and added complexity; omitted.
- JPNLOLITONOSTSAM.csv [japan_enrichment]: Useful OECD Japan leading indicator, but redundant with the top-level OECD file already selected.
- Historic-cape-ratios.csv inside zip [valuation]: Duplicate of top-level CAPE upload; kept out to avoid duplicate source paths.
- country_level_macro_data OECD BTS CSV [oecd_leading]: Large and broad survey file with many overlapping measures; useful for later follow-up, not first compact pass.

## Too Fragile
- china_labour_market.csv [china_macro]: Mostly annual and sparse; not suitable for monthly modeling extension.
- china_fiscal_socioeconomic.csv [china_macro]: Mixed annual/monthly/governance content with heavy sparsity and unclear release timing.
- japan_10y_bond_yield_cpi.csv [japan_enrichment]: Overlaps baseline Japan rate/CPI backbone and mixes daily/monthly coverage awkwardly.
- japan_10y_bond_topix_dividends.csv [japan_enrichment]: File contents appear inconsistent with file name; not reliable enough for automated use.
- China Price History_SSE_Composite.xlsx [china_sleeve_candidate]: Local index price history is not a USD adjusted-close investable sleeve target and does not support baseline-style target construction.
- japan buyback yield.xlsx [japan_enrichment]: Workbook appears to be price-history style export rather than clean buyback-yield time series.
- EU_IG_.xlsx [credit_stress]: Total-return price-history workbook, not a clean spread/yield indicator for compact experimental use.
