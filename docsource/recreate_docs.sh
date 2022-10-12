#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

# execute this inside docsource

rm -r source/*
rm -r ../docs/doctrees/*
rm -r ../docs/html/*

conda activate && sphinx-apidoc -f -o source/ ../src/quocslib/
conda activate && make github