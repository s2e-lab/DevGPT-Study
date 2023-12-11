def packet_handler(packet):
    if packet.haslayer(IP):
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        print(f"Received packet from {src_ip} to {dst_ip}.")

sniff(iface="eth0", prn=packet_handler, count=10)
