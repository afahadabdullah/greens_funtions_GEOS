#!/bin/bash
# Direct python execution bypassing conda command
PY_BIN="/nobackupp27/afahad/conda/envs/earth/bin/python"
export PATH="/nobackupp27/afahad/conda/envs/earth/bin:$PATH"

echo "Using Python: $PY_BIN"
$PY_BIN --version

cd /nobackupp27/afahad/GFEXPv11.7/process
echo "Starting Plotting on $(hostname) at $(date)"

# Run plot script
$PY_BIN -u plot_diff.py > log_plot.txt 2>&1

echo "Plotting finished at $(date)"
