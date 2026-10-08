"""Interactive load testing with the Locust web UI (http://localhost:8089).

    pip install locust && locust -f loadtest/locustfile.py --host http://localhost:8080

Locust is closed-loop (each user waits for its reply), which under-reports
latency once the server saturates. The numbers in the report come from the
open-loop generator in loadgen.py instead.
"""

import random
import sys
from pathlib import Path

from locust import HttpUser, constant_throughput, task

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from dataset import split  # noqa: E402

_, EVAL = split()
IMAGES = [(p.read_bytes(), label) for p, label in random.Random(0).sample(EVAL, 500)]


class Client(HttpUser):
    wait_time = constant_throughput(1)  # each user sends about 1 request per second

    @task
    def predict(self):
        body, label = random.choice(IMAGES)
        with self.client.post("/predict", data=body, name="/predict", catch_response=True) as r:
            if r.status_code == 200:
                r.success()
            else:
                r.failure(f"{r.status_code} {r.json().get('reason', '')}")
