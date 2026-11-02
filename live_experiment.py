"""
live_experiment.py
------------------
Runs the REAL socket programs (server.py + client.py) on localhost and plots
the results - proof that the algorithm works over actual UDP sockets, not
only inside the simulator.

For each timeout strategy a fresh lossy server is started:
    10 % packet loss, ~60 ms delay for the first 150 requests, then a
    "congestion" period with ~220 ms delay.

Outputs:
    results/live_<mode>.csv
    graphs/fig9_live_rtt_rto.png
    graphs/fig10_live_summary.png

Run:  python live_experiment.py        (takes about 4 minutes)
"""
import os
import subprocess
import sys
import time
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from client import run_client
import simulation as sim  # re-use styling and output folders

HERE = os.path.dirname(os.path.abspath(__file__))
N = 300
SERVER_ARGS = ["--loss", "0.10", "--delay", "0.06", "--jitter", "0.012",
               "--congest-after", "150", "--congest-delay", "0.22"]
RUNS = [  # (label, mode, fixed timeout, style key)
    ("Adaptive (Jacobson/Karels)", "adaptive", None, "Adaptive (Jacobson/Karels)"),
    ("EWMA (RFC 793)", "ewma", None, "EWMA (RFC 793)"),
    ("Fixed 100 ms", "fixed", 0.10, "Fixed 200 ms"),
    ("Fixed 1 s", "fixed", 1.0, "Fixed 1 s"),
]


def start_server(port):
    p = subprocess.Popen([sys.executable, os.path.join(HERE, "server.py"),
                          "--port", str(port)] + SERVER_ARGS,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)
    return p


def main():
    results = {}
    for i, (label, mode, ft, _) in enumerate(RUNS):
        port = 9990 + i
        srv = start_server(port)
        try:
            print(f"--- running {label} ---")
            rows, summ = run_client("127.0.0.1", port, mode, N,
                                    fixed_timeout=ft or 1.0, verbose=False,
                                    out=os.path.join(sim.RESULT_DIR, f"live_{mode}"
                                                     f"{'_' + str(int(ft*1000)) if ft else ''}.csv"))
            results[label] = (rows, summ)
            print("   ", summ)
        finally:
            srv.terminate()
            srv.wait()

    # ---- Fig 9: live RTT vs RTO
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.2), sharex=True, sharey=True)
    for ax, (label, mode, ft, key) in zip(axes.flat, RUNS):
        rows, _ = results[label]
        x = [r["seq"] for r in rows]
        rtt = [r["rtt_ms"] if r["rtt_ms"] != "" and r["attempts"] == 1 else None
               for r in rows]
        ax.scatter([a for a, b in zip(x, rtt) if b is not None],
                   [b for b in rtt if b is not None], s=7, color=sim.RTT_COLOR,
                   alpha=0.6, linewidths=0, label="Measured RTT")
        c, ls = sim.STYLE[key]
        ax.plot(x, [r["rto_ms"] for r in rows], color=c, ls=ls, label="RTO")
        ax.set_title(label, loc="left")
        ax.set_ylim(0, 1300)
        ax.legend(loc="upper left")
    for ax in axes[1]:
        ax.set_xlabel("Request number")
    for ax in axes[:, 0]:
        ax.set_ylabel("Time (ms)")
    fig.suptitle("Fig 9. Live UDP sockets on localhost: RTO vs RTT "
                 "(10 % loss; delay jumps 60 -> 220 ms after the server's 150th packet)",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(sim.GRAPH_DIR, "fig9_live_rtt_rto.png"))
    plt.close(fig)

    # ---- Fig 10: live summary
    labels = [r[0] for r in RUNS]
    short = ["Adaptive", "EWMA", "Fixed\n100 ms", "Fixed\n1 s"]
    colors = [sim.STYLE[r[3]][0] for r in RUNS]
    metrics = [("elapsed_s", "Total time for 300 requests (s)", "{:.1f}"),
               ("mean_latency_ms", "Mean latency (ms)", "{:.0f}"),
               ("retransmissions", "Retransmissions", "{:.0f}")]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    table = []
    for ax, (k, title, fmt) in zip(axes, metrics):
        vals = [results[l][1][k] for l in labels]
        bars = ax.bar(short, vals, color=colors, width=0.62,
                      edgecolor=sim.SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v),
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.15)
    fig.suptitle("Fig 10. Live socket experiment - summary", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(sim.GRAPH_DIR, "fig10_live_summary.png"))
    plt.close(fig)

    with open(os.path.join(sim.RESULT_DIR, "live_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["strategy"] + list(results[labels[0]][1]))
        w.writeheader()
        for l in labels:
            w.writerow({"strategy": l, **results[l][1]})
    print("\nLive graphs written to", sim.GRAPH_DIR)


if __name__ == "__main__":
    main()
