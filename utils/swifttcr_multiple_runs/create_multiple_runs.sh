#!/bin/bash

# original file
TEMPLATE="/home/wkrosschell/1_swifttcr/swifttcr/utils/swifttcr_multiple_runs/ensemble_paper_base.job"  
# target structure name   "7rm4"
DIRS=("7na5" "7pbc" "7pbe" "7pdw" "7phr" "7qpj" "7rk7" "7rm4" "7rrg" "8d5q" "8dnt" "8i5c" "8i5d" "8shi" "8wte" "8wul")

for d in "${DIRS[@]}"; do
    # make a safe filename for the new script
    NEWFILE="/$(basename "$d").sh"
    
    # copy the original file
    cp "$TEMPLATE" "$NEWFILE"
    
    # replace the hardcoded 'pdb_str' to target name
    sed -i "s|pdbName|$d|g" "$NEWFILE"
    
    echo "Created $NEWFILE with INPUT_DIR=$d"
done
