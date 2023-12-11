import subprocess
import threading

# Define the command to start the Minecraft server.
server_command = "java -Xmx2G -Xms2G -jar minecraft_server.jar nogui"

# Create a subprocess to start the server.
server_process = subprocess.Popen(server_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)

# Define a function to handle server output.
def handle_output():
    for line in server_process.stdout:
        print(line.strip())

# Start a thread to continuously read and print server output.
output_thread = threading.Thread(target=handle_output)
output_thread.start()

# Main loop to send commands to the server.
while True:
    command = input()
    server_process.stdin.write(command + "\n")
    server_process.stdin.flush()
