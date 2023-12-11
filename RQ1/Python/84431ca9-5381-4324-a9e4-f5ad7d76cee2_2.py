target_ip = "192.168.1.100"
target_port = 80

syn_packet = IP(dst=target_ip)/TCP(dport=target_port, flags="S")
response = sr1(syn_packet, timeout=2, verbose=False)

if response and response.haslayer(TCP) and response.getlayer(TCP).flags == "SA":
    print("TCP SYN scan successful! Port", target_port, "is open.")
else:
    print("TCP SYN scan failed! Port", target_port, "may be closed.")
