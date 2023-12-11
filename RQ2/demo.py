import os
import filecmp

# NOTE: This program will compare two directories to check if they contain the same files with the same content


def get_all_files(dir_path):
    """Recursively get all file paths in a directory."""
    file_paths = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def compare_directories(dir1, dir2):
    """Compare two directories to check if they contain the same files with the same content."""
    files1 = get_all_files(dir1)
    files2 = get_all_files(dir2)
    print(f"Found {len(files1)} files in {dir1}")
    print(f"Found {len(files2)} files in {dir2}")

    if len(files1) != len(files2):
        return False  # Different number of files

    # Convert to relative paths for easy comparison
    rel_files1 = {os.path.relpath(file, dir1) for file in files1}
    rel_files2 = {os.path.relpath(file, dir2) for file in files2}

    if rel_files1 != rel_files2:
        return False  # File names mismatch

    for file in rel_files1:
        if not filecmp.cmp(os.path.join(dir1, file), os.path.join(dir2, file), shallow=False):
            return False  # File contents mismatch

    return True


dir1 = '../results/Data_Java_Python_20230727'
dir2 = '../results/Data_Java_Python_20230817'
are_dirs_same = compare_directories(dir1, dir2)
print("Directories are the same." if are_dirs_same else "Directories are different.")
