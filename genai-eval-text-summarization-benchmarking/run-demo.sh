#!/bin/bash

source activate-conda.sh
activate_conda
conda activate qualbench

echo "Run demo to compare a candidate summary to a reference summary"
CANDIDATE=candidate_summarization.txt
REFERENCE=reference_summarization.txt
python run_qualitative_metrics.py -c $CANDIDATE -r $REFERENCE  
