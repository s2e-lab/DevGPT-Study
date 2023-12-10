class Negentropy:
    def __init__(self, idSize):
        self.idSize = idSize
        self.items = []  # to store the items (timestamp, id)
        self.sealed = False

    def addItem(self, timestamp, _id):
        if not self.sealed:
            if len(_id) != self.idSize:
                raise ValueError(f"ID size is not {self.idSize} bytes")
            self.items.append((timestamp, _id))
        else:
            raise RuntimeError("Can't add items after sealing the object")

    def seal(self):
        self.sealed = True

    def initiate(self):
        # The logic to create an initial message
        # This would use methods like fingerprinting and splitting
        # For simplification, let's return a placeholder
        return "INIT_MESSAGE"

    def reconcile(self, msg):
        if not self.sealed:
            raise RuntimeError("Object must be sealed before reconciliation")

        # Actual logic to process the received message and produce a response.
        # This would require functions to handle various range types (Skip, Fingerprint, IdList, IdListResponse)
        # For now, we'll assume a simple logic: if the message is "INIT_MESSAGE", we reply with an empty message.
        
        if msg == "INIT_MESSAGE":
            return ("", [], [])  # return response, have, need
        else:
            # Placeholder logic
            return ("RESPONSE", [], [])

