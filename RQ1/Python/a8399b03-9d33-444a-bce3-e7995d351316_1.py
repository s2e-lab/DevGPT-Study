IP = "10.6.6.88"
PORT = 1337

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    client_socket.connect((IP, PORT))
