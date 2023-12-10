server = Server()
server.start()

client = Client()
client.start()

# You may need to add logic here to keep the main thread alive, or to clean up the server/client when you're done.
