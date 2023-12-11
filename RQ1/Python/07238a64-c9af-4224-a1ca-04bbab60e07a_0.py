def should_ignore(file_path, ignore_list):
    for pattern in ignore_list:
        if pattern.endswith(os.sep):  # This is a directory pattern
            dir_pattern = pattern.rstrip(os.sep)
            if file_path.startswith(dir_pattern + os.sep):
                return True
        elif fnmatch.fnmatch(file_path, pattern):
            return True
    return False
