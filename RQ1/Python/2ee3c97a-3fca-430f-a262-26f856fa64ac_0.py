import socket

def create_server_socket(ip, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((ip, port))
    return server_socket

def receive_message(server_socket):
    data, client_address = server_socket.recvfrom(1024)
