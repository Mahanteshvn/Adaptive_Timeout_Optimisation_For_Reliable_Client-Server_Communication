"""
client.py
---------
Reliable UDP client (stop-and-wait ARQ) whose retransmission timeout is
chosen by a pluggable estimator from rto_estimator.py.

For every request:
  1. send "REQ:<seq>"
  2. wait up to RTO seconds for "ACK:REQ:<seq>"
  3. on timeout -> estimator.on_timeout() (back-off) and retransmit
  4. on ACK     -> if the packet was sent only ONCE, feed the RTT to the
                   estimator (Karn's algorithm), then move to next seq

Results are written to a CSV file for plotting (see plot_live.py).

Usage:
    python client.py --mode adaptive -n 200 --out results_adaptive.csv
    python client.py --mode fixed --fixed-timeout 0.1 -n 200
"""
import argparse
import csv
import socket
import time

from rto_estimator import make_estimator


def run_client(host, port, mode, n, fixed_timeout=1.0, max_retries=8, out=None,
               verbose=True, on_row=None, stop_event=None):
    """on_row(row) is called after every request (used by the UI for live charts)."""
    kw = {"timeout": fixed_timeout} if mode == "fixed" else {}
    est = make_estimator(mode, **kw)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rows = []
    t_start = time.perf_counter()
    total_tx = 0
    failures = 0

    for seq in range(n):
        if stop_event is not None and stop_event.is_set():
            n = seq
            break
        payload = f"REQ:{seq}".encode()
        expected = b"ACK:" + payload
        attempts = 0
        first_send = time.perf_counter()
        rtt = None
        while attempts <= max_retries:
            rto = est.rto()
            sock.settimeout(rto)
            send_t = time.perf_counter()
            sock.sendto(payload, (host, port))
            attempts += 1
            total_tx += 1
            try:
                while True:  # skip stale (late) ACKs of earlier sequences
                    data, _ = sock.recvfrom(2048)
                    if data == expected:
                        break
                    remaining = rto - (time.perf_counter() - send_t)
                    if remaining <= 0:
                        raise socket.timeout
                    sock.settimeout(remaining)
                rtt = time.perf_counter() - send_t
                if attempts == 1:           # Karn's algorithm
                    est.on_rtt_sample(rtt)
                break
            except socket.timeout:
                est.on_timeout()
        done = time.perf_counter()
        if rtt is None:
            failures += 1
        rows.append({
            "seq": seq,
            "attempts": attempts,
            "rtt_ms": round(rtt * 1000, 3) if rtt is not None else "",
            "rto_ms": round(est.rto() * 1000, 3),
            "latency_ms": round((done - first_send) * 1000, 3),
            "delivered": int(rtt is not None),
        })
        if on_row is not None:
            on_row(rows[-1])
        if verbose and seq % 20 == 0:
            print(f"[{mode}] seq={seq:4d} attempts={attempts} "
                  f"rtt={rows[-1]['rtt_ms']}ms rto={rows[-1]['rto_ms']}ms")

    elapsed = time.perf_counter() - t_start
    sock.close()

    summary = {
        "mode": mode, "requests": n, "transmissions": total_tx,
        "retransmissions": total_tx - n, "failures": failures,
        "elapsed_s": round(elapsed, 3),
        "mean_latency_ms": round(sum(r["latency_ms"] for r in rows) / max(n, 1), 2),
    }
    if out and rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows, summary


def main():
    ap = argparse.ArgumentParser(description="Reliable UDP client with adaptive RTO")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--mode", choices=["fixed", "ewma", "adaptive"], default="adaptive")
    ap.add_argument("-n", type=int, default=200, help="number of requests")
    ap.add_argument("--fixed-timeout", type=float, default=1.0)
    ap.add_argument("--max-retries", type=int, default=8)
    ap.add_argument("--out", default=None, help="CSV output file")
    args = ap.parse_args()

    _, s = run_client(args.host, args.port, args.mode, args.n,
                      args.fixed_timeout, args.max_retries, args.out)
    print("\n=== SUMMARY ===")
    for k, v in s.items():
        print(f"{k:18s}: {v}")


if __name__ == "__main__":
    main()
