import lnk
with open('file.None.0xfffffa8003bfa080.dat', 'rb') as f:
    link = lnk.parse(f.read())
print(link)
