#!/bin/bash
# Direct python execution bypassing conda command
PY_BIN="/nobackupp27/afahad/conda/envs/earth/bin/python"
export PATH="/nobackupp27/afahad/conda/envs/earth/bin:$PATH"

echo "Using Python: $PY_BIN"
$PY_BIN --version

cd /nobackupp27/afahad/GFEXPv11.7/process
echo "Starting execution on $(hostname) at $(date)"

# Run with unbuffered output
$PY_BIN -u extract_clim.py > log_extract.txt 2>&1

echo "Execution finished at $(date)"
