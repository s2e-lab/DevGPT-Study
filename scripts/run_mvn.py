#!/usr/bin/env python3

import subprocess
import os
import time
import csv

maven_project_path = "/Users/lsiddiqsunny/Documents/Notre_Dame/Research/DevGPT-Study/Java_maven"

tested_projects = set()  # Initialize an empty set to keep track of tested projects
output = []  # Accumulates all results

for root, dirs, files in os.walk(maven_project_path):
    for project in dirs:
        project_path = os.path.join(root, project)
        print(f"Testing {project_path}")


        # Check if the directory starts with 'ignore-' or if it's already been tested
        if project.startswith("ignore-") or project in tested_projects:
            print(f"Skipping {project}")
            continue

        if not os.path.exists(project_path):
            print(f"Directory {project_path} does not exist.")
            continue

        try:
            # Change directory to the project path
            os.chdir(project_path)

            # Run 'mvn test' command
            process = subprocess.Popen(['mvn', 'test'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process_output, process_error = process.communicate(timeout=45)

            # Handle output or errors if needed
            print(process_output.decode('utf-8'))

            # Add the tested project to the set
            tested_projects.add(project)

            # Parse the compilation output for errors (example)
            # Modify this part based on your actual error parsing logic
            lines = process_output.decode('utf-8').split('\n')
            for line in lines:
                if line.startswith("[ERROR]"):
                    output.append([project, line])  # Adjust output format as needed
            print(f"Finished testing {project_path}")

            print(output)
        except Exception as e:
            print(f"Error occurred: {e}")

# Save the compilation status to a CSV file
absFilePath = os.path.abspath(__file__)
os.chdir(os.path.dirname(absFilePath))
out_file = f"output_{time.time()}.csv"

with open(out_file, 'w', newline='', encoding='UTF8') as csvfile:
    csv_writer = csv.writer(csvfile)
    for row in output:
        csv_writer.writerow(row)

print("SUCCESS: Saved all", len(output), "errors in", out_file)

