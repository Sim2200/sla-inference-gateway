"""Model server: one ONNX image classifier behind a small FastAPI app.

POST /predict takes raw image bytes and returns the top-5 ImageNet classes.

Concurrency model: at most MAX_CONCURRENCY inferences run at once (default 1, so
each request gets all ORT_THREADS cores); other requests wait in a queue. When
more than MAX_QUEUE requests are waiting, new ones get 503 right away instead of
waiting forever. That backpressure is what lets the gateway see overload early.

Environment
-----------
MODEL_DIR            directory with model.onnx and meta.json            (required)
LABELS_PATH          JSON list of 1,000 class names       (MODEL_DIR/../labels.json)
TIER, VERSION        labels reported in responses and metrics
ORT_THREADS          ONNX Runtime intra-op threads                             (2)
MAX_CONCURRENCY      inferences allowed at the same time                        (1)
MAX_QUEUE            waiting requests before 503                               (64)
FAULT_ERROR_RATE     fraction of requests that fail with 500, for testing     (0.0)
FAULT_LATENCY_MS     extra latency added to each request, for testing           (0)
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import anyio.to_thread
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.concurrency import run_in_threadpool

from .preprocess import preprocess_bytes

MODEL_DIR = Path(os.environ["MODEL_DIR"])
LABELS_PATH = Path(os.environ.get("LABELS_PATH", MODEL_DIR.parent / "labels.json"))
TIER = os.environ.get("TIER", "unknown")
VERSION = os.environ.get("VERSION", "v1")
ORT_THREADS = int(os.environ.get("ORT_THREADS", "2"))
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "1"))
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "64"))
FAULT_ERROR_RATE = float(os.environ.get("FAULT_ERROR_RATE", "0"))
FAULT_LATENCY_MS = float(os.environ.get("FAULT_LATENCY_MS", "0"))

META = json.loads((MODEL_DIR / "meta.json").read_text())
LABELS = json.loads(LABELS_PATH.read_text())
MODEL_NAME = META["name"]

BUCKETS = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10)
LABELNAMES = ("tier", "version", "model")
REQUESTS = Counter("modelserver_requests_total", "Requests by outcome", (*LABELNAMES, "outcome"))
LATENCY = Histogram("modelserver_request_seconds", "Time in server, including queueing", LABELNAMES, buckets=BUCKETS)
INFERENCE = Histogram("modelserver_inference_seconds", "Preprocess + ONNX Runtime time", LABELNAMES, buckets=BUCKETS)
IN_FLIGHT = Gauge("modelserver_in_flight", "Requests queued or running", LABELNAMES)
LABEL_VALUES = (TIER, VERSION, MODEL_NAME)


def _session() -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = ORT_THREADS
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(MODEL_DIR / "model.onnx"), opts, providers=["CPUExecutionProvider"])


SESSION = _session()
SLOTS = threading.Semaphore(MAX_CONCURRENCY)
_lock = threading.Lock()
_in_flight = 0
_ready = False


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _ready
    # Every waiting request holds a worker thread, so size the pool to the queue.
    anyio.to_thread.current_default_thread_limiter().total_tokens = MAX_QUEUE + MAX_CONCURRENCY + 8
    x = np.zeros((1, 3, META["crop"], META["crop"]), dtype=np.float32)
    for _ in range(5):
        SESSION.run(None, {"input": x})
    _ready = True
    yield


app = FastAPI(title=f"model server ({TIER}/{MODEL_NAME})", lifespan=lifespan)


def _softmax_top5(logits: np.ndarray) -> list[dict]:
    z = logits - logits.max()
    p = np.exp(z) / np.exp(z).sum()
    top = np.argsort(-p)[:5]
    return [{"index": int(i), "label": LABELS[i], "score": round(float(p[i]), 4)} for i in top]


def _enter() -> None:
    global _in_flight
    with _lock:
        if _in_flight >= MAX_QUEUE + MAX_CONCURRENCY:
            REQUESTS.labels(*LABEL_VALUES, "rejected").inc()
            raise HTTPException(503, "queue full", headers={"Retry-After": "1"})
        _in_flight += 1
        IN_FLIGHT.labels(*LABEL_VALUES).set(_in_flight)


def _leave() -> None:
    global _in_flight
    with _lock:
        _in_flight -= 1
        IN_FLIGHT.labels(*LABEL_VALUES).set(_in_flight)


@app.post("/predict")
async def predict(request: Request) -> dict:
    body = await request.body()
    if not body:
        raise HTTPException(400, "send the image bytes as the request body")
    # The heavy part runs in a worker thread so the event loop stays responsive.
    return await run_in_threadpool(_predict_sync, body)


def _predict_sync(body: bytes) -> dict:
    start = time.perf_counter()
    _enter()
    try:
        with SLOTS:
            queued = time.perf_counter()
            if FAULT_LATENCY_MS:
                time.sleep(FAULT_LATENCY_MS / 1000)
            if FAULT_ERROR_RATE and random.random() < FAULT_ERROR_RATE:
                REQUESTS.labels(*LABEL_VALUES, "error").inc()
                raise HTTPException(500, "injected fault")
            try:
                x = preprocess_bytes(body, META["resize"], META["crop"])
            except Exception as exc:  # not an image
                REQUESTS.labels(*LABEL_VALUES, "bad_request").inc()
                raise HTTPException(400, f"could not decode image: {exc}") from exc
            logits = SESSION.run(None, {"input": x})[0][0]
            done = time.perf_counter()
        INFERENCE.labels(*LABEL_VALUES).observe(done - queued)
        LATENCY.labels(*LABEL_VALUES).observe(done - start)
        REQUESTS.labels(*LABEL_VALUES, "ok").inc()
        return {
            "tier": TIER, "version": VERSION, "model": MODEL_NAME,
            "top5": _softmax_top5(logits),
            "queue_ms": round((queued - start) * 1000, 2),
            "inference_ms": round((done - queued) * 1000, 2),
        }
    finally:
        _leave()


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict:
    if not _ready:
        raise HTTPException(503, "warming up")
    return {"status": "ready"}


@app.get("/info")
def info() -> dict:
    return {"tier": TIER, "version": VERSION, "meta": META, "ort_threads": ORT_THREADS,
            "max_concurrency": MAX_CONCURRENCY, "max_queue": MAX_QUEUE, "in_flight": _in_flight,
            "fault_error_rate": FAULT_ERROR_RATE, "fault_latency_ms": FAULT_LATENCY_MS}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
