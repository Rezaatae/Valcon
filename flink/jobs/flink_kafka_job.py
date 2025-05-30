from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream.connectors import FlinkKafkaConsumer
from pyflink.common.typeinfo import Types
import json

def main():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)

    # Define Kafka source
    kafka_source = FlinkKafkaConsumer(
        topics='quickstart-events',
        deserialization_schema=SimpleStringSchema(),
        properties={
            'bootstrap.servers': 'localhost:9092',
            'group.id': 'flink-test-group',
            'auto.offset.reset': 'earliest'
        }
    )

    # Create a stream from Kafka
    data_stream = env.add_source(kafka_source).map(
        lambda msg: json.loads(msg),
        output_type=Types.MAP(Types.STRING(), Types.STRING())
    )

    # Print the data
    data_stream.print()

    # Execute the Flink job
    env.execute("Flink Kafka Consumer Python Job")

if __name__ == '__main__':
    main()
