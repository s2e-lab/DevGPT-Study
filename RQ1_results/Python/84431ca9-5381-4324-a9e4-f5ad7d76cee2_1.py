from scapy.all import *

target_ip = "192.168.1.1"
icmp_packet = IP(dst=target_ip)/ICMP()
response = sr1(icmp_packet, timeout=2, verbose=False)

if response:
    print("Ping successful! Response time:", response.time, "ms")
else:
    print("Ping failed! The target may be unreachable.")
