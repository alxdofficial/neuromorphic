"""Shared embedding service/client for lightweight A-MEM orchestration workers.

The server is the only process that imports SentenceTransformers/PyTorch.  Workers use
``RemoteSentenceTransformer`` as a drop-in subset of the upstream API and retain their own A-MEM note,
link, and embedding-index state.  Vectors travel as base64 float32 rather than enormous JSON number lists.
"""
from __future__ import annotations

import base64
import json
import os
import queue
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

PROTOCOL_VERSION = 1


@dataclass
class _BatchRequest:
    sentences: list[str]
    done: threading.Event = field(default_factory=threading.Event)
    result: np.ndarray | None = None
    error: BaseException | None = None


class EmbeddingBatcher:
    """Serialize GPU access while coalescing concurrent worker requests into short dynamic batches."""

    def __init__(self, encoder, *, batch_size: int = 128, max_wait_ms: float = 5.0):
        self.encoder = encoder
        self.batch_size = batch_size
        self.max_wait_s = max_wait_ms / 1000.0
        self.queue: queue.Queue[_BatchRequest | None] = queue.Queue()
        self._stats_lock = threading.Lock()
        self._stats = {"batches": 0, "requests": 0, "sentences": 0,
                       "max_batch_sentences": 0, "failures": 0}
        self.thread = threading.Thread(target=self._run, name="amem-embedding-batcher", daemon=True)
        self.thread.start()

    def encode(self, sentences: list[str]) -> np.ndarray:
        if not sentences:
            return np.empty((0, 0), dtype=np.float32)
        request = _BatchRequest(sentences)
        self.queue.put(request)
        request.done.wait()
        if request.error is not None:
            raise RuntimeError(f"embedding batch failed: {request.error}") from request.error
        assert request.result is not None
        return request.result

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join(timeout=10)

    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)

    def _run(self) -> None:
        while True:
            first = self.queue.get()
            if first is None:
                return
            requests = [first]
            n_sentences = len(first.sentences)
            deadline = time.monotonic() + self.max_wait_s
            while n_sentences < self.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    nxt = self.queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if nxt is None:
                    self.queue.put(None)
                    break
                requests.append(nxt)
                n_sentences += len(nxt.sentences)
            flat = [sentence for request in requests for sentence in request.sentences]
            with self._stats_lock:
                self._stats["batches"] += 1
                self._stats["requests"] += len(requests)
                self._stats["sentences"] += len(flat)
                self._stats["max_batch_sentences"] = max(self._stats["max_batch_sentences"], len(flat))
            try:
                vectors = self.encoder.encode(
                    flat, batch_size=self.batch_size, convert_to_numpy=True,
                    normalize_embeddings=False, show_progress_bar=False,
                )
                vectors = np.ascontiguousarray(vectors, dtype=np.float32)
                if vectors.ndim != 2 or vectors.shape[0] != len(flat):
                    raise ValueError(f"encoder returned shape {vectors.shape} for {len(flat)} sentences")
                offset = 0
                for request in requests:
                    end = offset + len(request.sentences)
                    request.result = vectors[offset:end]
                    offset = end
            except BaseException as exc:  # propagate one batch failure to every waiting HTTP request
                with self._stats_lock:
                    self._stats["failures"] += 1
                for request in requests:
                    request.error = exc
            finally:
                for request in requests:
                    request.done.set()


def _pack_vectors(vectors: np.ndarray) -> dict[str, Any]:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    return {
        "protocol": PROTOCOL_VERSION,
        "dtype": "float32",
        "shape": list(vectors.shape),
        "data": base64.b64encode(vectors.tobytes()).decode("ascii"),
    }


def _unpack_vectors(payload: dict[str, Any]) -> np.ndarray:
    if payload.get("protocol") != PROTOCOL_VERSION or payload.get("dtype") != "float32":
        raise ValueError("embedding service protocol/dtype mismatch")
    shape = payload.get("shape")
    if not isinstance(shape, list) or len(shape) != 2 or not all(type(x) is int and x >= 0 for x in shape):
        raise ValueError(f"invalid embedding shape {shape!r}")
    raw = base64.b64decode(payload.get("data", ""), validate=True)
    expected = shape[0] * shape[1] * np.dtype(np.float32).itemsize
    if len(raw) != expected:
        raise ValueError(f"embedding payload has {len(raw)} bytes, expected {expected}")
    return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()


class _EmbeddingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, *, batcher: EmbeddingBatcher, health: dict[str, Any]):
        super().__init__(address, handler)
        self.batcher = batcher
        self.health = health


class EmbeddingRequestHandler(BaseHTTPRequestHandler):
    server: _EmbeddingHTTPServer
    max_body_bytes = 16 * 1024 * 1024

    def log_message(self, fmt: str, *args) -> None:
        if os.environ.get("AMEM_EMBEDDING_HTTP_LOG") == "1":
            super().log_message(fmt, *args)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path != "/health":
            self._json(404, {"error": "not found"})
            return
        self._json(200, {**self.server.health, "stats": self.server.batcher.stats()})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path != "/encode":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > self.max_body_bytes:
                raise ValueError(f"invalid request size {length}")
            payload = json.loads(self.rfile.read(length))
            sentences = payload.get("sentences")
            if not isinstance(sentences, list) or not sentences or not all(isinstance(x, str) for x in sentences):
                raise ValueError("sentences must be a non-empty list of strings")
            vectors = self.server.batcher.encode(sentences)
            self._json(200, _pack_vectors(vectors))
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": f"{type(exc).__name__}: {exc}"})
        except Exception as exc:  # keep the server alive; client retry policy decides terminal behavior
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})


def serve_embeddings(*, model_name: str, device: str, host: str, port: int,
                     batch_size: int = 128, max_wait_ms: float = 5.0) -> None:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    probe = np.asarray(model.encode(["A-MEM embedding service health probe"],
                                   convert_to_numpy=True, show_progress_bar=False))
    health = {
        "ok": True,
        "protocol": PROTOCOL_VERSION,
        "model": model_name,
        "device": device,
        "dimension": int(probe.shape[1]),
        "batch_size": batch_size,
        "max_wait_ms": max_wait_ms,
    }
    batcher = EmbeddingBatcher(model, batch_size=batch_size, max_wait_ms=max_wait_ms)
    server = _EmbeddingHTTPServer((host, port), EmbeddingRequestHandler, batcher=batcher, health=health)
    print(f"[amem-embedding-service] ready http://{host}:{port} model={model_name} device={device} "
          f"dim={health['dimension']} batch_size={batch_size}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("[amem-embedding-service] stopped", flush=True)
    finally:
        server.server_close()
        batcher.close()


class RemoteSentenceTransformer:
    """The small SentenceTransformer surface used by upstream ``SimpleEmbeddingRetriever``."""

    def __init__(self, model_name: str, *, base_url: str | None = None, timeout: float = 60.0,
                 retries: int = 5, **_kwargs):
        self.model_name = model_name
        self.base_url = (base_url or os.environ.get("AMEM_EMBEDDING_URL") or "").rstrip("/")
        if not self.base_url:
            raise ValueError("AMEM_EMBEDDING_URL / base_url is required for remote embeddings")
        self.timeout = timeout
        self.retries = retries

    def get_config_dict(self) -> dict[str, str]:
        return {"model_name": self.model_name}

    def health(self) -> dict[str, Any]:
        with urllib.request.urlopen(f"{self.base_url}/health", timeout=self.timeout) as response:
            payload = json.loads(response.read())
        if not payload.get("ok") or payload.get("protocol") != PROTOCOL_VERSION:
            raise RuntimeError(f"unhealthy/incompatible embedding service: {payload}")
        return payload

    def encode(self, sentences, *, convert_to_tensor: bool = False, **_kwargs):
        if convert_to_tensor:
            raise ValueError("remote A-MEM embedding client returns numpy arrays, not torch tensors")
        single = isinstance(sentences, str)
        values = [sentences] if single else list(sentences)
        body = json.dumps({"sentences": values}, separators=(",", ":")).encode()
        request = urllib.request.Request(
            f"{self.base_url}/encode", data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    vectors = _unpack_vectors(json.loads(response.read()))
                return vectors[0] if single else vectors
            except (OSError, ValueError, urllib.error.HTTPError, urllib.error.URLError) as exc:
                if attempt + 1 == self.retries:
                    raise RuntimeError(f"embedding service failed after {self.retries} attempts: {exc}") from exc
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")


class _UnusedDependency:
    def __init__(self, *_args, **_kwargs):
        raise RuntimeError("this unused upstream dependency is unavailable in lightweight A-MEM mode")


def _cosine_similarity(x, y) -> np.ndarray:
    x_arr, y_arr = np.asarray(x), np.asarray(y)
    dots = x_arr @ y_arr.T
    denom = np.linalg.norm(x_arr, axis=1)[:, None] * np.linalg.norm(y_arr, axis=1)[None, :]
    return np.divide(dots, denom, out=np.zeros_like(dots), where=denom != 0)


def install_lightweight_upstream_imports(service_url: str) -> None:
    """Install only the symbols imported by upstream A-MEM, before importing ``memory_layer``.

    The evaluated AgenticMemorySystem/MemoryNote implementation still comes directly from the pinned upstream
    checkout.  Heavy modules used only by its alternate retrievers/backends are replaced with explicit stubs;
    attempting to select one fails loudly rather than silently changing behavior.
    """
    if "memory_layer" in sys.modules:
        raise RuntimeError("install lightweight A-MEM imports before importing upstream memory_layer")
    os.environ["AMEM_EMBEDDING_URL"] = service_url.rstrip("/")

    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = RemoteSentenceTransformer
    transformers = types.ModuleType("transformers")
    transformers.AutoModel = _UnusedDependency
    transformers.AutoTokenizer = _UnusedDependency
    litellm = types.ModuleType("litellm")
    litellm.completion = lambda *_a, **_k: (_ for _ in ()).throw(
        RuntimeError("LiteLLM backend is unavailable in lightweight A-MEM mode"))
    rank_bm25 = types.ModuleType("rank_bm25")
    rank_bm25.BM25Okapi = _UnusedDependency
    sklearn = types.ModuleType("sklearn")
    sklearn.__path__ = []
    sklearn_metrics = types.ModuleType("sklearn.metrics")
    sklearn_metrics.__path__ = []
    sklearn_pairwise = types.ModuleType("sklearn.metrics.pairwise")
    sklearn_pairwise.cosine_similarity = _cosine_similarity
    nltk = types.ModuleType("nltk")
    nltk.__path__ = []
    nltk_tokenize = types.ModuleType("nltk.tokenize")
    nltk_tokenize.word_tokenize = lambda text: text.split()

    modules = {
        "sentence_transformers": sentence_transformers,
        "transformers": transformers,
        "litellm": litellm,
        "rank_bm25": rank_bm25,
        "sklearn": sklearn,
        "sklearn.metrics": sklearn_metrics,
        "sklearn.metrics.pairwise": sklearn_pairwise,
        "nltk": nltk,
        "nltk.tokenize": nltk_tokenize,
    }
    sys.modules.update(modules)
