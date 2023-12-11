from flask import Flask, jsonify
import pika

app = Flask(__name__)
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Queue names
request_queue = 'request_queue'
response_queue = 'response_queue'

# Initialize response mapping
responses = {}

# Endpoint to handle requests
@app.route('/api/endpoint', methods=['POST'])
def handle_request():
    # Generate a unique identifier for the request
    request_id = generate_unique_id()

    # Send message to request queue
    channel.basic_publish(exchange='', routing_key=request_queue, body=request_id)

    # Wait for the response
    response = wait_for_response(request_id)

    if response:
        return jsonify(response)
    else:
        return jsonify({'error': 'Timeout occurred'}), 500

# Function to wait for the response
def wait_for_response(request_id):
    # Start a timer for the timeout
    start_time = time.time()
    timeout = 0.35  # Timeout duration in seconds

    # Wait for the response until timeout occurs
    while time.time() - start_time < timeout:
        # Check if the response is available
        if request_id in responses:
            return responses.pop(request_id)

        # Sleep for a short interval before checking again
        time.sleep(0.01)

    return None

# Function to handle incoming messages from RabbitMQ
def handle_message(channel, method, properties, body):
    # Store the response in the responses dictionary
    request_id = body.decode()
    responses[request_id] = {'message': 'Your response'}

    # Acknowledge the message
    channel.basic_ack(delivery_tag=method.delivery_tag)

# Set up RabbitMQ consumer
channel.queue_declare(queue=response_queue)
channel.basic_consume(queue=response_queue, on_message_callback=handle_message)

if __name__ == '__main__':
    app.run(debug=True)
