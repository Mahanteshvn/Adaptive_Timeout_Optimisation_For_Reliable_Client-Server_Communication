"""
app.py  -  Adaptive Timeout Lab (web UI)
=======================================
Runs a small local web server and opens the project's dashboard in your
browser.  Uses ONLY the Python standard library - no pip install needed.

    python3 app.py            # then open http://localhost:8000 (opens automatically)
    python3 app.py --port 8080 --no-browser

Tabs in the UI
    1. Overview         - the problem, the four strategies, the formulas
    2. Simulation lab   - change RTT / jitter / loss / scenario and see every
                          strategy's RTO track the network, with a scorecard
    3. Experiments      - interactive versions of the report's sweeps
                          (loss, jitter, LAN->satellite, fixed-timeout dilemma, outage)
    4. Live sockets     - starts REAL UDP servers and clients on this computer
                          and draws RTT/RTO live as packets flow
    5. Report graphs    - the matplotlib figures from simulation.py

API (JSON)
    POST /api/simulate        one scenario, every strategy
    POST /api/sweep           a parameter sweep (kind = loss|jitter|network|fixed|outage)
    POST /api/live/start      start live socket runs, returns {"id": ...}
    GET  /api/live/status     ?id=..&since=..  rows so far for each strategy
    POST /api/live/stop       {"id": ...}
"""
import argparse
import json
import math
import os
import threading
import time
import uuid
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import server as lossy_server
from client import run_client
from rto_estimator import FixedTimeout, JacobsonKarels
from sim_core import (Network, simulate, averaged, make_strategy, STRATEGY_NAMES,
                      JacobsonNoBackoff)

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
GRAPHS = os.path.join(HERE, "graphs")


def clean(x):
    """JSON-safe number (NaN -> None), rounded."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(x, 4) if isinstance(x, float) else x


def labels_for(fs_ms, fl_ms):
    """Display names of the four strategies using the chosen fixed values."""
    return {
        "Adaptive (Jacobson/Karels)": "Adaptive (Jacobson/Karels)",
        "EWMA (RFC 793)": "EWMA (RFC 793)",
        "Fixed 200 ms": f"Fixed {fs_ms:g} ms",
        "Fixed 1 s": f"Fixed {fl_ms:g} ms" if fl_ms < 1000 else f"Fixed {fl_ms / 1000:g} s",
    }


# ------------------------------------------------------------- simulation
def build_network(p, seed):
    base = p.get("base_rtt_ms", 100) / 1000
    jit = p.get("jitter_ms", 10) / 1000
    loss = p.get("loss_pct", 5) / 100
    n = int(p.get("n", 600))
    spikes = 0.02 if p.get("spikes", True) else 0.0
    scen = p.get("scenario", "congestion")
    outage = None
    if scen == "congestion":
        mult = p.get("congestion_factor", 3.0)
        phases = [(0, base, jit), (n // 3, base * mult, jit * mult * 2),
                  (2 * n // 3, base * 0.8, jit * 0.8)]
    elif scen == "outage":
        phases = [(0, base, jit)]
        outage = (p.get("outage_start_s", 5.0),
                  p.get("outage_start_s", 5.0) + p.get("outage_len_s", 10.0))
    else:
        phases = [(0, base, jit)]
    return Network(phases, loss=loss, spike_p=spikes, outage=outage,
                   min_rtt=0.001, seed=seed), phases, outage


def api_simulate(p):
    n = max(20, min(int(p.get("n", 600)), 3000))
    p["n"] = n
    seed = int(p.get("seed", 7))
    fs = p.get("fixed_short_ms", 200)
    fl = p.get("fixed_long_ms", 1000)
    names = labels_for(fs, fl)
    out = {"strategies": [], "phases": None, "outage": None}
    for key in STRATEGY_NAMES:
        net, phases, outage = build_network(p, seed)   # same seed = same network
        net.send_log = []
        est = make_strategy(key, fs / 1000, fl / 1000)
        recs, s = simulate(est, net, n, max_tx=20)
        out["phases"] = [{"start": a, "base_ms": b * 1000, "jitter_ms": c * 1000}
                         for a, b, c in phases]
        out["outage"] = outage
        out["strategies"].append({
            "key": key, "name": names[key],
            "rtt": [clean(r["rtt_s"] * 1000) for r in recs],
            "rto": [clean(r["rto_s"] * 1000) for r in recs],
            "latency": [clean(r["latency_s"] * 1000) for r in recs],
            "time": [clean(r["time_s"]) for r in recs],
            "attempts": [r["attempts"] for r in recs],
            "sends": [clean(t) for t in net.send_log],
            "summary": {k: clean(float(v)) for k, v in s.items()},
        })
    return out


def api_sweep(p):
    kind = p.get("kind", "loss")
    seeds = range(int(p.get("seeds", 6)))
    fs = p.get("fixed_short_ms", 200)
    fl = p.get("fixed_long_ms", 1000)
    names = labels_for(fs, fl)
    metrics = ["mean_latency_ms", "p95_latency_ms", "spurious_per_100", "efficiency_pct",
               "latency_overhead"]

    def summarise(s, n, base=None):
        d = {"mean_latency_ms": s["mean_latency_ms"], "p95_latency_ms": s["p95_latency_ms"],
             "spurious_per_100": 100 * s["spurious"] / n,
             "efficiency_pct": 100 * s["efficiency"]}
        if base:
            d["latency_overhead"] = s["mean_latency_ms"] / (1000 * base)
        return {k: round(v, 3) for k, v in d.items()}

    if kind == "outage":
        outage = (5.0, 5.0 + p.get("outage_len_s", 10.0))
        res = []
        for label, est in (("Adaptive + exponential back-off", JacobsonKarels()),
                           ("Adaptive, no back-off", JacobsonNoBackoff())):
            net = Network([(0, 0.100, 0.010)], loss=0.0, spike_p=0.0, outage=outage, seed=3)
            net.send_log = []
            simulate(est, net, 200, max_tx=30)
            sends = sorted(net.send_log)
            in_out = sum(outage[0] <= t < outage[1] for t in sends)
            res.append({"name": label, "sends": [round(t, 3) for t in sends],
                        "in_outage": in_out})
        return {"kind": kind, "outage": outage, "series": res}

    if kind == "fixed":
        n = 600
        phases = [(0, 0.100, 0.010), (200, 0.300, 0.060), (400, 0.080, 0.008)]
        xs = [100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000]
        fixed = []
        for t in xs:
            s = averaged(lambda sd: Network(phases, loss=0.05, seed=sd),
                         lambda: FixedTimeout(t / 1000), n, seeds)
            fixed.append(summarise(s, n))
        a = averaged(lambda sd: Network(phases, loss=0.05, seed=sd),
                     lambda: JacobsonKarels(), n, seeds)
        return {"kind": kind, "x": xs, "fixed": fixed, "adaptive": summarise(a, n),
                "metrics": metrics}

    n = 400
    if kind == "loss":
        xs = [0, 2, 5, 10, 15, 20, 25, 30]
        mk = lambda x, sd: Network([(0, 0.120, 0.030)], loss=x / 100, seed=sd)
        xlabels = [f"{x}%" for x in xs]
        bases = [None] * len(xs)
    elif kind == "jitter":
        xs = [5, 20, 40, 60, 80, 100, 120]
        mk = lambda x, sd: Network([(0, 0.150, x / 1000)], loss=0.02, spike_p=0.0, seed=sd)
        xlabels = [f"{x} ms" for x in xs]
        bases = [None] * len(xs)
    elif kind == "network":
        nets = [("LAN", 5), ("Metro", 30), ("National", 80), ("Intercontinental", 200),
                ("Mobile 3G", 400), ("Satellite", 600)]
        xs = [b for _, b in nets]
        xlabels = [f"{l} ({b} ms)" for l, b in nets]
        mk = lambda x, sd: Network([(0, x / 1000, 0.15 * x / 1000)], loss=0.05,
                                   spike_p=0.0, min_rtt=0.001, seed=sd)
        bases = [x / 1000 for x in xs]
    else:
        raise ValueError("unknown sweep kind")

    series = []
    for key in STRATEGY_NAMES:
        vals = []
        for x, base in zip(xs, bases):
            s = averaged(lambda sd: mk(x, sd), lambda: make_strategy(key, fs / 1000, fl / 1000),
                         n, seeds)
            vals.append(summarise(s, n, base))
        series.append({"key": key, "name": names[key], "values": vals})
    return {"kind": kind, "x": xs, "xlabels": xlabels, "series": series, "metrics": metrics}


# ------------------------------------------------------------ live sockets
LIVE = {}
LIVE_LOCK = threading.Lock()


class LiveRun:
    def __init__(self, p):
        self.id = uuid.uuid4().hex[:10]
        self.p = p
        self.stop = threading.Event()
        self.started = time.time()
        self.lanes = []
        n = max(10, min(int(p.get("n", 200)), 1000))
        for spec in p.get("strategies", []):
            lane = {"label": spec["label"], "key": spec.get("key", spec["label"]),
                    "mode": spec["mode"], "fixed": spec.get("fixed_timeout_ms", 1000) / 1000,
                    "rows": [], "summary": None, "server": {}, "error": None,
                    "n": n, "t0": None}
            self.lanes.append(lane)
        for lane in self.lanes:
            threading.Thread(target=self._run_lane, args=(lane,), daemon=True).start()

    def _run_lane(self, lane):
        p = self.p
        srv_stop = threading.Event()
        ready = threading.Event()
        info = lane["server"]
        threading.Thread(target=lossy_server.serve, daemon=True, kwargs=dict(
            port=0, loss=p.get("loss_pct", 10) / 100, delay=p.get("delay_ms", 60) / 1000,
            jitter=p.get("jitter_ms", 12) / 1000, congest_after=int(p.get("congest_after", 150)),
            congest_delay=p.get("congest_delay_ms", 220) / 1000, stop_event=srv_stop,
            verbose=False, ready_event=ready, info=info)).start()
        ready.wait(3)
        lane["t0"] = time.time()

        def on_row(r):
            r = dict(r)
            r["t"] = round(time.time() - lane["t0"], 3)
            lane["rows"].append(r)
        try:
            _, s = run_client("127.0.0.1", info["port"], lane["mode"], lane["n"],
                              fixed_timeout=lane["fixed"], verbose=False,
                              on_row=on_row, stop_event=self.stop)
            lane["summary"] = s
        except Exception as e:  # show errors in the UI instead of dying silently
            lane["error"] = repr(e)
        finally:
            srv_stop.set()

    def status(self, since):
        lanes = []
        for lane in self.lanes:
            rows = lane["rows"]
            lanes.append({
                "label": lane["label"], "key": lane["key"], "n": lane["n"],
                "rows": rows[since:], "count": len(rows),
                "done": lane["summary"] is not None or lane["error"] is not None,
                "summary": lane["summary"], "error": lane["error"],
                "server_packets": lane["server"].get("packets", 0),
                "server_dropped": lane["server"].get("dropped", 0),
                "port": lane["server"].get("port"),
                "elapsed": (lane["summary"]["elapsed_s"] if lane["summary"]
                            else round(time.time() - lane["t0"], 2) if lane["t0"] else 0),
            })
        return {"id": self.id, "lanes": lanes,
                "done": all(l["done"] for l in lanes)}


# ------------------------------------------------------------------ server
MIME = {".html": "text/html; charset=utf-8", ".js": "application/javascript",
        ".css": "text/css", ".png": "image/png", ".svg": "image/svg+xml",
        ".csv": "text/csv"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # keep the terminal quiet

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, root, rel):
        path = os.path.normpath(os.path.join(root, rel))
        if not path.startswith(root) or not os.path.isfile(path):
            return self._send(404, {"error": "not found"})
        with open(path, "rb") as f:
            data = f.read()
        self._send(200, data, MIME.get(os.path.splitext(path)[1], "application/octet-stream"))

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._file(STATIC, "index.html")
        if u.path.startswith("/static/"):
            return self._file(STATIC, u.path[len("/static/"):])
        if u.path.startswith("/graphs/"):
            return self._file(GRAPHS, u.path[len("/graphs/"):])
        if u.path == "/api/graphs":
            files = sorted(f for f in os.listdir(GRAPHS) if f.endswith(".png")) \
                if os.path.isdir(GRAPHS) else []
            return self._send(200, {"files": files})
        if u.path == "/api/live/status":
            q = parse_qs(u.query)
            run = LIVE.get(q.get("id", [""])[0])
            if not run:
                return self._send(404, {"error": "unknown run"})
            return self._send(200, run.status(int(q.get("since", ["0"])[0])))
        self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        try:
            p = json.loads(self.rfile.read(length) or b"{}")
            if u.path == "/api/simulate":
                return self._send(200, api_simulate(p))
            if u.path == "/api/sweep":
                return self._send(200, api_sweep(p))
            if u.path == "/api/live/start":
                with LIVE_LOCK:
                    for r in LIVE.values():     # only one live run at a time
                        r.stop.set()
                    run = LiveRun(p)
                    LIVE.clear()
                    LIVE[run.id] = run
                return self._send(200, {"id": run.id})
            if u.path == "/api/live/stop":
                run = LIVE.get(p.get("id"))
                if run:
                    run.stop.set()
                return self._send(200, {"ok": True})
        except Exception as e:
            return self._send(500, {"error": repr(e)})
        self._send(404, {"error": "not found"})


def main():
    ap = argparse.ArgumentParser(description="Adaptive Timeout Lab - web UI")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    httpd = None
    for port in range(a.port, a.port + 20):      # find a free port
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if httpd is None:
        raise SystemExit("No free port found")
    url = f"http://localhost:{httpd.server_address[1]}"
    print("=" * 60)
    print("  Adaptive Timeout Lab is running")
    print(f"  Open {url} in your browser")
    print("  Press Ctrl+C here to stop")
    print("=" * 60)
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
