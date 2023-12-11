class EventStream:
    def __init__(self):
        self.listeners = []

    def emit(self, event):
        for listener in self.listeners:
            listener(event)

    def subscribe(self, listener):
        self.listeners.append(listener)

# Instantiate event stream
events = EventStream()

# Define a listener
def print_event(event):
    print(f'Received event: {event}')

# Subscribe the listener to the event stream
events.subscribe(print_event)

# Emit some events
events.emit('Event 1')
events.emit('Event 2')
