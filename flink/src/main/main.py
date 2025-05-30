import os
from dotenv import load_dotenv
import subprocess

# Load .env
load_dotenv()

FLINK_HOME = os.getenv("FLINK_HOME")
JAR_PATH = os.getenv("KAFKA_CONNECTOR_JAR")
JOB_PATH = os.path.abspath("/Users/reza/Valcon/flink/jobs/flink_kafka_job.py")

if not FLINK_HOME or not JAR_PATH:
    raise RuntimeError("FLINK_HOME or KAFKA_CONNECTOR_JAR not set in .env")

flink_run_cmd = [
    os.path.join(FLINK_HOME, "bin", "flink"),
    "run",
    "--python", JOB_PATH,
    "--jarfile", JAR_PATH
]

print("Running Flink job...")
subprocess.run(flink_run_cmd)
