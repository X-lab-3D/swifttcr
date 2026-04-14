#!/bin/bash

# ====================================================================================================
# Script: create_multiple_runs.sh
# Purpose: Copies a given script and changes a given word in the script to a list of other 
# words. Originally designed to create ensemble_paper_base.job scripts for multiple cases. Output 
# files are placed in current directory.
# Author: Wieke Krösschell
# Usage: bash create_multiple_runs.sh
# ====================================================================================================

# Original file that needs to be copied
TEMPLATE="/utils/swifttcr_multiple_runs/ensemble_paper_base.job"  

# All cases that need a script to run SwiftTCR
DIRS=("7na5" "7pbc" "7pbe" "7pdw" "7phr" "7qpj" "7rk7" "7rm4" "7rrg" "8d5q" "8dnt" "8i5c" "8i5d" "8shi" "8wte" "8wul")

for d in "${DIRS[@]}"; do
    # mMkes a safe filename for the new script
    NEWFILE="/$(basename "$d").sh"
    
    # Copies the original file
    cp "$TEMPLATE" "$NEWFILE"
    
    # Replaces the hardcoded 'pdbName' to target name
    sed -i "s|pdbName|$d|g" "$NEWFILE"
    
    echo "Created $NEWFILE with INPUT_DIR=$d"
done
