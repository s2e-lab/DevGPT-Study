from scapy.all import *

def process_packet(packet):
    if IP in packet:
        ip = packet[IP]
        if TCP in packet:
            port = packet[TCP].sport
        elif UDP in packet:
            port = packet[UDP].sport
        else:
            return

        ip_address = ip.src
        print(f"IP: {ip_address} Port: {port}")

# Specify the path to your PCAP file
pcap_file = "path/to/your/file.pcap"

# Load the PCAP file
packets = rdpcap(pcap_file)

# Process each packet in the PCAP file
for packet in packets:
    process_packet(packet)
