#!/bin/bash
python generate_sbatch.py $1 $2 $3 $4 $5;
sbatch $2;
rm $2;
