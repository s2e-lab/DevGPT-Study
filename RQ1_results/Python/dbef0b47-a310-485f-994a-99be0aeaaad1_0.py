import socket

def start_dcc_server():
    """
    Start a DCC server that listens for a connection.
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('', 0))  # Bind to any available port
    server_socket.listen(1)

    host, port = server_socket.getsockname()
    print(f"Listening on {host}:{port}")

    # Wait for a connection
    client_socket, addr = server_socket.accept()
    print(f"Accepted connection from {addr}")

    # You can now use client_socket to send and receive data

def send_dcc_request(irc_socket, target, host, port):
    """
    Send a DCC request to the given target over the given IRC socket.
    """
    # Format the DCC request
    # The \x01 character is used to denote the start and end of the CTCP message
    message = f"\x01DCC CHAT chat {socket.inet_aton(host).hex()} {port}\x01"
    irc_socket.send(f"PRIVMSG {target} :{message}\r\n".encode())

def handle_dcc_request(irc_socket, message):
    """
    Handle a DCC request received over the given IRC socket.
    """
    # Extract the host and port from the message
    _, _, _, encoded_host, port = message.split()
    host = socket.inet_ntoa(bytes.fromhex(encoded_host))

    print(f"Received DCC request from {host}:{port}")

    # Connect to the DCC server
    dcc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    dcc_socket.connect((host, int(port)))

    # You can now use dcc_socket to send and receive data
