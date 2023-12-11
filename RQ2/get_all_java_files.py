import os


def find_java_files(directory, output_file_paths, output_file_names):
    with open(output_file_paths, 'w') as file_paths, open(output_file_names, 'w') as file_names:
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith('.java'):
                    file_path = os.path.join(root, filename)
                    file_paths.write(file_path + '\n')
                    file_names.write(filename + '\n')


project_directory = './Repo'
output_filename_paths = 'java_files_full_paths.txt'
output_filename_names = 'java_files_names.txt'
find_java_files(project_directory, output_filename_paths,
                output_filename_names)
