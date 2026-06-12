'''Inference Worker for MAL-ZDA Framework
This script defines the Inference Worker component of the MAL-ZDA framework.
The worker consumes network telemetry data from a Kafka topic, processes it, and forwards it to a FastAPI microservice for anomaly detection.
It handles the integration between the data ingestion layer and the inference engine, ensuring that incoming data is correctly formatted 
and that results are logged for monitoring and analysis.
The worker is designed to be lightweight and efficient, suitable for deployment in edge environments where real-time processing is critical.    
    '''

import json
import requests
from confluent_kafka import Consumer, KafkaError

def run_inference_worker():
    # Configure Consumer to read from the local broker
    consumer = Consumer({
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'malzda-inference-group',
        'auto.offset.reset': 'earliest'
    })

    topic = 'network-telemetry'
    consumer.subscribe([topic])
    
    api_endpoint = "http://127.0.0.1:8000/api/v1/detect"

    print(f"Worker connected. Listening to topic: {topic}")
    print(f"Forwarding to inference engine at: {api_endpoint}")

    try:
        while True:
            msg = consumer.poll(1.0)

            if msg is None:
                continue
            if msg.error():
                err = msg.error()
                if err and err.code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(f"Consumer error: {msg.error()}")
                    break

            # Parse the Kafka message
            value = msg.value()
            if value:
                raw_data = json.loads(value.decode('utf-8'))
            else:
                continue
            
            # Prepare the exact payload expected by the FastAPI service
            api_payload = {
                "packet_features": raw_data["packet_features"],
                "flow_features": raw_data["flow_features"],
                "campaign_features": raw_data["campaign_features"]
            }

            # Hit the microservice
            try:
                response = requests.post(api_endpoint, json=api_payload)
                if response.status_code == 200:
                    result = response.json()
                    print(f"[{raw_data['client_id']}] Anomaly Score: {result['anomaly_score']:.3f} | Zero-Day: {result['is_zero_day_candidate']} | Latency: {result['processing_time_ms']}ms")
                else:
                    print(f"API Error: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                print("Connection failed. Is the FastAPI microservice running on port 8000?")

    except KeyboardInterrupt:
        print("Worker shutting down.")
    finally:
        consumer.close()

if __name__ == "__main__":
    run_inference_worker()