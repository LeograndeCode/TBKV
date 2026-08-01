#!/usr/bin/env bash
# Regenerate the two figures the paper includes.
#
# The paper has exactly two \includegraphics, both produced by a single script:
#   fig_vid672_frontier.pdf   Figure 1 (fig:frontier)
#   fig_vid672_ablation.pdf   Figure 2 (fig:ablation)
#
# Fast (seconds, CPU-only): the script carries the plotted values as verified
# constants, so it runs no model and needs no GPU, weights, dataset or prior
# step -- it works on a bare checkout of this archive.
#
# Output lands in paper/Figures/, which is where \graphicspath resolves
# \includegraphics from. A .png is written alongside each .pdf.
#
# Usage:  bash scripts/run/06_figures.sh

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

banner "Regenerating paper figures -> paper/Figures/"

run python scripts/plots/make_vitdet672_figures.py

banner "Done"
cat <<'EOF'
  Regenerated in paper/Figures/:
    fig_vid672_frontier.{pdf,png}   Figure 1  (fig:frontier)
    fig_vid672_ablation.{pdf,png}   Figure 2  (fig:ablation)

  These are the only two figures in the paper, and this is the only plotting
  script in the archive. It needs no GPU, no weights and no dataset.

  NOTE: make_vitdet672_figures.py carries its plotted values as constants
  rather than reading the result directories, so re-running an evaluation does
  not change the figure until those constants are updated.
EOF
