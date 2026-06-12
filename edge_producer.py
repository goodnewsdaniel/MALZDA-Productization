'''Edge Producer Script
This script simulates an edge agent that extracts features from network flows
and sends them to a Kafka topic for further processing. 
It generates random data to mimic real network telemetry 
and continuously streams it to the Kafka cluster.'''


import json
import time
import random
import numpy as np
from confluent_kafka import Producer

def delivery_report(err, msg):
    """ Called once for each message produced to indicate delivery result. """
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Telemetry queued to {msg.topic()} partition [{msg.partition()}]")

def simulate_network_traffic():
    # Configure the Kafka Producer to point to your local Docker container
    producer = Producer({'bootstrap.servers': 'localhost:9092'})
    topic = 'network-telemetry'

    print("Edge Agent initialized. Streaming network telemetry...")

    try:
        while True:
            # Simulate the extraction of features from a network flow
            # We inject slight randomness to simulate live traffic variations
            payload = {
                "client_id": "tenant_maziv_01",
                "timestamp": time.time(),
                "packet_features": np.random.randn(256).tolist(),
                "flow_features": np.random.randn(100, 256).tolist(),
                "campaign_features": np.random.randn(1024).tolist()
            }

            # Asynchronously push to Kafka
            producer.produce(
                topic, 
                value=json.dumps(payload).encode('utf-8'), 
                callback=delivery_report
            )
            
            producer.poll(0)
            
            # Simulate a new network flow every 2 seconds
            time.sleep(2)

    except KeyboardInterrupt:
        print("Agent shutting down.")
    finally:
        # Wait for any outstanding messages to be delivered
        producer.flush()

if __name__ == "__main__":
    simulate_network_traffic()