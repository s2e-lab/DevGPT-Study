#!/bin/bash

PROJECT_DIR="../../results/RQ2_Commit_Tree"

OUTPUT_FILE="pylint_output.json"

if ! command -v jq &> /dev/null; then
    echo "[{\"error\": \"jq not found, please install it.\"}]" > $OUTPUT_FILE
    exit 1
fi

# Temporary file to store individual pylint outputs
TEMP_OUTPUT="temp_pylint_output.json"

# Start JSON array
echo "[" > $TEMP_OUTPUT

# Find all Python files and run pylint on each
first=true
find "$PROJECT_DIR" -type f -name "*.py" | while read -r file; do
    pylint_output=$(pylint --output-format=json "$file" 2>&1)
    pylint_status=$?

    # Separate JSON objects with a comma, except for the first entry
    if $first; then
        first=false
    else
        echo "," >> $TEMP_OUTPUT
    fi

    if [ $pylint_status -ne 0 ]; then
        # Handle pylint error
        echo "{\"file\": \"$file\", \"error\": \"Pylint error\", \"details\": $(echo $pylint_output | jq -aRs .)}" >> $TEMP_OUTPUT
    else
        # Handle successful pylint output
        echo "$pylint_output" | jq -s --arg filename "$file" '{($filename): .}' >> $TEMP_OUTPUT
    fi
done

# Close JSON array and move to final output
echo "]" >> $TEMP_OUTPUT
mv $TEMP_OUTPUT $OUTPUT_FILE
