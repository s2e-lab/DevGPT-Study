import socket
import struct
import time

def traceroute(host, max_hops=30):
    port = 33434
    ttl = 1
    while True:
        # Crearea unui socket de tip UDP și setarea TTL-ului acestuia
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.setsockopt(socket.IPPROTO_IP, socket.IP_TTL, struct.pack('I', ttl))

        # Începerea cronometrării și trimiterea unui pachet de date către destinație
        start_time = time.time()
        udp_socket.sendto(b"", (host, port))

        # Așteptarea unui răspuns de la destinație sau de la un ruter intermediar
        try:
            data, address = udp_socket.recvfrom(1024)
            end_time = time.time()
            elapsed_time = round((end_time - start_time) * 1000, 2)
            print(f"{ttl}. {address[0]} ({elapsed_time} ms)")
        except socket.error:
            print(f"{ttl}. *")

        # Închiderea socket-ului de tip UDP
        udp_socket.close()

        # Dacă am ajuns la destinație sau la numărul maxim de salturi, ieșim din buclă
        if address[0] == host or ttl >= max_hops:
            break

        # Trecerea la următorul TTL
        ttl += 1

if __name__ == "__main__":
    traceroute("google.com")
