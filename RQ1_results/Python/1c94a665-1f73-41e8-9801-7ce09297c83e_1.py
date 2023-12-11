import lnkfile

def parse_lnk_file(lnk_file_path):
    lnk = lnkfile.LnkFile(lnk_file_path)
    print("Target Path:", lnk.target_file)
    print("Command Arguments:", lnk.command_arguments)
    print("Working Directory:", lnk.working_directory)
    print("Icon Location:", lnk.icon_location)
    # Add more attributes as needed

# Replace 'file.496.0xfffffa80022ac740.resume.pdf.lnk.dat' with the actual path to your .lnk file
parse_lnk_file('file.496.0xfffffa80022ac740.resume.pdf.lnk.dat')
