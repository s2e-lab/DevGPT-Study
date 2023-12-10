import os


def count_python_files(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                count += 1
    return count


directory_path = '../../results/RQ2_Commit_Tree'
python_files_count = count_python_files(directory_path)
print(f"Number of Python files: {python_files_count}")

# NOTE: 568
