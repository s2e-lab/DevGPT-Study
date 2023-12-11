# Client Side
ne_client = Negentropy(16)
for item in [("timestamp1", "id1"), ("timestamp2", "id2")]:  # sample items
    ne_client.addItem(*item)
ne_client.seal()
msg = ne_client.initiate()
while len(msg) != 0:
    response = "RESPONSE_FROM_SERVER"  # replace with actual function to query server
    msg, have, need = ne_client.reconcile(response)
    # handle have/need

# Server Side
ne_server = Negentropy(16)
for item in [("timestamp1", "id1"), ("timestamp3", "id3")]:  # sample items
    ne_server.addItem(*item)
ne_server.seal()
while True:
    msg = "MESSAGE_FROM_CLIENT"  # replace with function to receive from client
    response, _, _ = ne_server.reconcile(msg)
    # send response to client
