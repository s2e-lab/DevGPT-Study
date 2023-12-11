from scapy.all import *

def process_packet(packet):
    if IP in packet:
        ip = packet[IP]
        if TCP in packet:
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
        elif UDP in packet:
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
        else:
            return

        src_ip = ip.src
        dst_ip = ip.dst
        print(f"Source IP: {src_ip} Port: {src_port}")
        print(f"Destination IP: {dst_ip} Port: {dst_port}")

# Specify the path to your PCAP file
pcap_file = "path/to/your/file.pcap"

# Load the PCAP file
packets = rdpcap(pcap_file)

# Process each packet in the PCAP file
for packet in packets:
    process_packet(packet)
