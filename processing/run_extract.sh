#!/bin/bash
# Explicit conda initialization
CONDA_SH="/nobackup/gmao_SIteam/GEOSpyD/4.11.0_py3.9/2022-04-28/etc/profile.d/conda.sh"

if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
else
    echo "CRITICAL: Conda script not found at $CONDA_SH"
    exit 1
fi

conda activate earth
echo "Environment activated: $(which python)"
echo "Python Version: $(python --version)"

cd /nobackupp27/afahad/GFEXPv11.7/process
echo "Starting execution on $(hostname) at $(date)"

# Run with unbuffered output
python -u extract_clim.py > log_extract.txt 2>&1

echo "Execution finished at $(date)"
