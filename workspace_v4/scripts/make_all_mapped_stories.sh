#!/bin/bash

cd /Users/batuhanatas/Desktop/XOPTPOE/workspace_v4

mkdir -p reports/v4_mapped_stories

echo "Generating story for 2020-12-31 | Q7_stretch_excess..."
PYTHONPATH=src python reports/make_beautiful_conference_story.py \
  --workspace . \
  --scenario_csv reports/scenario_results_mapped_v4.csv \
  --anchor 2020-12-31 \
  --question_id Q7_stretch_excess \
  --output reports/v4_mapped_stories/story_2020_Q7_stretch_excess.png

echo "Generating story for 2021-12-31 | Q1_more_gold..."
PYTHONPATH=src python reports/make_beautiful_conference_story.py \
  --workspace . \
  --scenario_csv reports/scenario_results_mapped_v4.csv \
  --anchor 2021-12-31 \
  --question_id Q1_more_gold \
  --output reports/v4_mapped_stories/story_2021_Q1_more_gold.png

echo "Generating story for 2021-12-31 | Q5_more_diversified..."
PYTHONPATH=src python reports/make_beautiful_conference_story.py \
  --workspace . \
  --scenario_csv reports/scenario_results_mapped_v4.csv \
  --anchor 2021-12-31 \
  --question_id Q5_more_diversified \
  --output reports/v4_mapped_stories/story_2021_Q5_more_diversified.png

echo "Generating story for 2022-12-31 | Q6_classic_60_40..."
PYTHONPATH=src python reports/make_beautiful_conference_story.py \
  --workspace . \
  --scenario_csv reports/scenario_results_mapped_v4.csv \
  --anchor 2022-12-31 \
  --question_id Q6_classic_60_40 \
  --output reports/v4_mapped_stories/story_2022_Q6_classic_60_40.png

echo "Generating story for 2024-12-31 | Q4_less_gold..."
PYTHONPATH=src python reports/make_beautiful_conference_story.py \
  --workspace . \
  --scenario_csv reports/scenario_results_mapped_v4.csv \
  --anchor 2024-12-31 \
  --question_id Q4_less_gold \
  --output reports/v4_mapped_stories/story_2024_Q4_less_gold.png

echo "Generating story for 2026-02-28 | Q11_max_sharpe_total..."
PYTHONPATH=src python reports/make_beautiful_conference_story.py \
  --workspace . \
  --scenario_csv reports/scenario_results_mapped_v4.csv \
  --anchor 2026-02-28 \
  --question_id Q11_max_sharpe_total \
  --output reports/v4_mapped_stories/story_2026_Q11_max_sharpe_total.png

echo "All mapped stories have been successfully saved to workspace_v4/reports/v4_mapped_stories/"
