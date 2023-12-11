target_ip = "www.example.com"
traceroute_packet = IP(dst=target_ip, ttl=(1, 30))/ICMP()
traceroute_response, _ = sr(traceroute_packet, verbose=False)

for _, packet in traceroute_response:
    if packet[ICMP].type == 11:
        print("Hop", packet[IP].ttl, ":", packet[IP].src)
    elif packet[ICMP].type == 0:
        print("Reached destination:", packet[IP].src)
        break
