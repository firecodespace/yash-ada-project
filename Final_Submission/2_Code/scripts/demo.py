"""Local demo web app: classify syllogisms with the trained model and compare with the exact algorithm.

    python scripts/demo.py            # then open http://127.0.0.1:8000

The model is loaded once at start-up, so each check afterwards takes milliseconds.
No extra dependencies: it uses Python's built-in HTTP server.
"""

import argparse
import json
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from gen_synthetic import ALL_MOODS, LEAD, TEMPLATES, make, pool  # noqa: E402
from src.logic import verdict  # noqa: E402
from train_lora import PROMPT  # noqa: E402

LOCK = threading.Lock()
STATE = {}


def load(adapter: Path, base: str):
    tok = AutoTokenizer.from_pretrained(adapter)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2, torch_dtype=dtype)
    model.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(model, adapter).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    STATE.update(tok=tok, model=model)
    classify(["warm up. all a are b. all b are c. therefore all a are c."])  # first call is slower


def classify(texts):
    tok, model = STATE["tok"], STATE["model"]
    enc = tok([PROMPT.format(s=t) for t in texts], return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
    with LOCK, torch.inference_mode():
        p = model(**enc).logits.float().softmax(-1)[:, 1].tolist()
    return [{"valid": x > 0.5, "p_valid": round(x, 4)} for x in p]


def build_text(figure, mood, s, m, p, rnd):
    pairs = {1: [(m, p), (s, m)], 2: [(p, m), (s, m)], 3: [(m, p), (m, s)], 4: [(p, m), (m, s)]}[figure]
    prem = [rnd.choice(TEMPLATES[mood[i]]).format(s=a, p=b) for i, (a, b) in enumerate(pairs)]
    concl = rnd.choice(TEMPLATES[mood[2]]).format(s=s, p=p)
    lead = rnd.choice(LEAD)
    return " ".join(prem + [lead + (concl[0].lower() + concl[1:] if lead else concl)])


def api(path, body):
    t0 = time.perf_counter()
    if path == "/api/predict":
        text = body.get("text", "").strip()
        if not text:
            return {"error": "Enter a syllogism."}
        r = classify([text])[0]
    elif path == "/api/build":
        figure, mood = int(body["figure"]), body["mood"].upper()
        if figure not in (1, 2, 3, 4) or mood not in ALL_MOODS:
            return {"error": "Bad figure or mood."}
        terms = [body.get(k, "").strip() or d for k, d in (("s", "cats"), ("m", "mammals"), ("p", "animals"))]
        text = build_text(figure, mood, *terms, random.Random(body.get("seed")))
        r = classify([text])[0]
        r.update(text=text, exact=verdict(figure, mood))
    elif path == "/api/random":
        n = min(int(body.get("n", 30)), 100)
        rnd, words, rows = random.Random(), pool(True), []
        for i in range(n):
            ex = make(rnd, words, i % 2 == 0, True)
            rows.append(ex)
        preds = classify([e["syllogism"] for e in rows])
        ok = sum(p["valid"] == e["validity"] for p, e in zip(preds, rows))
        wrong = [{"text": e["syllogism"], "truth": e["validity"], "model": p["valid"]}
                 for p, e in zip(preds, rows) if p["valid"] != e["validity"]][:5]
        r = {"n": n, "correct": ok, "wrong": wrong}
    else:
        return {"error": "not found"}
    r["ms"] = round((time.perf_counter() - t0) * 1000)
    return r


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (ROOT / "scripts" / "demo.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/meta":
            self._send(200, json.dumps({"moods": ALL_MOODS}), "application/json")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        try:
            out = api(self.path, body)
        except Exception as exc:  # keep the demo alive on bad input
            out = {"error": str(exc)}
        self._send(200, json.dumps(out), "application/json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, default=ROOT / "outputs/lora-qwen15b-syn/adapter")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    print("Loading model (one time)...", flush=True)
    load(a.adapter, a.base)
    print(f"Ready: http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
