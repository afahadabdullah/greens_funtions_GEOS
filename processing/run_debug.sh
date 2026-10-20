#!/bin/bash
# Redirect stdout/stderr to a debug log in the same dir
LOGFILE="/nobackupp27/afahad/GFEXPv11.7/process/debug_run.log"
exec > "$LOGFILE" 2>&1

echo "=== Starting Run at $(date) ==="
echo "Hostname: $(hostname)"
echo "User: $(whoami)"

CONDA_SH="/nobackup/gmao_SIteam/GEOSpyD/4.11.0_py3.9/2022-04-28/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
    echo "Sourcing $CONDA_SH"
    source "$CONDA_SH"
else
    echo "CRITICAL: Conda script missing!"
    exit 1
fi

echo "--- Conda Info ---"
conda info --envs

echo "--- Activating Earth ---"
conda activate earth
ACT_STATUS=$?
echo "Activation Status: $ACT_STATUS"

echo "--- Python Debug ---"
echo "Which python: $(which python)"
python --version
echo "PATH: $PATH"

cd /nobackupp27/afahad/GFEXPv11.7/process
echo "--- Running Script ---"
python -u extract_clim.py > log_extract.txt 2>&1
SCRIPT_STATUS=$?
echo "Script finished with exit code $SCRIPT_STATUS at $(date)"
