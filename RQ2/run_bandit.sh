#!/bin/bash

cd ../../results/RQ2_Commit_Tree

# Find all Python files and run Bandit on them, outputting the results in JSON format
find . -name '*.py' | xargs bandit -f json -o results.json