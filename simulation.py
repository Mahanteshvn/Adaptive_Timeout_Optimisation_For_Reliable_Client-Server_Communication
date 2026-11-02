"""
simulation.py
-------------
Discrete-event simulation of a stop-and-wait client talking to a server
over an unreliable network, comparing four timeout strategies:

    Fixed 200 ms   - aggressive fixed timeout
    Fixed 1 s      - conservative fixed timeout
    EWMA (RFC 793) - smoothed mean only, RTO = 2 * SRTT
    Adaptive       - Jacobson/Karels (RFC 6298) + Karn + exponential back-off

Network model (per transmission):
    * lost with probability p (request or reply)
    * otherwise RTT ~ max(min_rtt, Normal(base_rtt(t), jitter(t))) plus
      occasional delay spikes (heavy tail, like queue build-ups)

A late ACK of an earlier transmission still completes the request (the
client accepts the first matching ACK) - so a timeout that fires while the
original ACK is still in flight produces a *spurious* retransmission.

Run:  python simulation.py           -> writes graphs/*.png and results/*.csv
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rto_estimator import FixedTimeout, JacobsonKarels
from sim_core import (Network, simulate, averaged, strategies, PHASES,
                      JacobsonNoBackoff)

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR = os.path.join(HERE, "graphs")
RESULT_DIR = os.path.join(HERE, "results")
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ----------------------------------------------------------------- styling
SURFACE = "#fcfcfb"
TEXT1, TEXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {   # colour (validated categorical order) + line style (secondary cue)
    "Adaptive (Jacobson/Karels)": ("#2a78d6", "-"),
    "EWMA (RFC 793)":             ("#eb6834", "--"),
    "Fixed 200 ms":               ("#1baf7a", "-."),
    "Fixed 1 s":                  ("#eda100", ":"),
}
RTT_COLOR = "#8a8984"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT1,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2, "savefig.dpi": 150,
})


# ------------------------------------------------------------- experiments
def save_csv(name, rows):
    with open(os.path.join(RESULT_DIR, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def legend_below(ax, ncol=2):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=ncol)


def exp1_rto_tracking():
    """Graph 1: how each RTO follows a changing RTT."""
    print("Experiment 1: RTO tracking across congestion phases")
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), sharex=True, sharey=True)
    for ax, (name, est) in zip(axes.flat, strategies().items()):
        net = Network(PHASES, loss=0.03, seed=7)
        recs, _ = simulate(est, net, 900)
        save_csv(f"exp1_trace_{name.split()[0].lower()}_{name.split()[1].lower()}.csv"
                 if name.startswith("Fixed") else
                 f"exp1_trace_{name.split()[0].lower()}.csv", recs)
        x = [r["req"] for r in recs]
        ax.scatter(x, [1000 * r["rtt_s"] for r in recs], s=6, color=RTT_COLOR,
                   alpha=0.55, label="Measured RTT", linewidths=0)
        c, ls = STYLE[name]
        ax.plot(x, [1000 * r["rto_s"] for r in recs], color=c, ls=ls, label="RTO")
        for p in (300, 600):
            ax.axvline(p, color=TEXT2, lw=0.8, ls=(0, (2, 3)))
        ax.set_title(name, loc="left")
        ax.set_ylim(0, 1300)
    for ax in axes[1]:
        ax.set_xlabel("Request number")
    for ax in axes[:, 0]:
        ax.set_ylabel("Time (ms)")
    for ax in axes[0]:
        for xm, lab in ((150, "normal"), (450, "congestion"), (750, "fast path")):
            ax.text(xm, 1240, lab, ha="center", color=TEXT2, fontsize=9)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="", marker="o", ms=4, color=RTT_COLOR,
                      label="Measured RTT (one dot per request)")]
    handles += [Line2D([], [], color=c, ls=ls, label=f"RTO - {n}")
                for n, (c, ls) in STYLE.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Fig 1. RTO vs measured RTT as network conditions change",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(GRAPH_DIR, "fig1_rto_tracking.png"))
    plt.close(fig)


def exp2_loss_sweep():
    """Graphs 2 & 3: latency and retransmissions vs packet-loss rate."""
    print("Experiment 2: packet-loss sweep")
    losses = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    phases = [(0, 0.120, 0.030)]
    table = []
    res = {name: [] for name in STYLE}
    for p in losses:
        for name in STYLE:
            s = averaged(lambda sd: Network(phases, loss=p, seed=sd),
                         lambda: strategies()[name], 500)
            res[name].append(s)
            table.append({"loss": p, "strategy": name,
                          **{k: round(v, 4) for k, v in s.items()}})
    save_csv("exp2_loss_sweep.csv", table)

    xs = [100 * p for p in losses]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for name, (c, ls) in STYLE.items():
        ax.plot(xs, [s["mean_latency_ms"] for s in res[name]], color=c, ls=ls,
                marker="o", ms=5, label=name)
    ax.set_xlabel("Packet loss rate (%)")
    ax.set_ylabel("Mean request latency (ms)")
    ax.set_title("Fig 2. Mean latency vs packet loss", loc="left")
    legend_below(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPH_DIR, "fig2_latency_vs_loss.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for name, (c, ls) in STYLE.items():
        ax.plot(xs, [100 * s["efficiency"] for s in res[name]], color=c, ls=ls,
                marker="o", ms=5, label=name)
    ax.plot(xs, [100 - x for x in xs], color=TEXT2, lw=1, ls=(0, (1, 2)),
            label="Ideal (only real losses resent)")
    ax.set_xlabel("Packet loss rate (%)")
    ax.set_ylabel("Transmission efficiency (%)\nrequests / packets sent")
    ax.set_title("Fig 3. Transmission efficiency vs packet loss", loc="left")
    legend_below(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPH_DIR, "fig3_efficiency_vs_loss.png"))
    plt.close(fig)
    return res, losses


def exp3_fixed_timeout_tradeoff():
    """Graph 4: no single fixed timeout is right - the U-shaped trade-off."""
    print("Experiment 3: fixed-timeout trade-off")
    phases = PHASES
    timeouts = [0.10, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
    lat, spur = [], []
    rows = []
    for t in timeouts:
        s = averaged(lambda sd: Network(phases, loss=0.05, seed=sd),
                     lambda: FixedTimeout(t), 900)
        lat.append(s["mean_latency_ms"])
        spur.append(s["spurious"] / 9)
        rows.append({"timeout_s": t, **{k: round(v, 4) for k, v in s.items()}})
    a = averaged(lambda sd: Network(phases, loss=0.05, seed=sd),
                 lambda: JacobsonKarels(), 900)
    rows.append({"timeout_s": "adaptive", **{k: round(v, 4) for k, v in a.items()}})
    save_csv("exp3_fixed_tradeoff.csv", rows)

    xs = [1000 * t for t in timeouts]
    cA = STYLE["Adaptive (Jacobson/Karels)"][0]
    cF = STYLE["Fixed 200 ms"][0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    ax1.plot(xs, lat, color=cF, marker="o", ms=5, label="Fixed timeout")
    ax1.axhline(a["mean_latency_ms"], color=cA, ls="--", label="Adaptive (Jacobson/Karels)")
    ax1.set_xscale("log")
    ax1.set_xlabel("Fixed timeout value (ms, log scale)")
    ax1.set_ylabel("Mean request latency (ms)")
    ax1.set_title("(a) Latency", loc="left")
    best = int(np.argmin(lat))
    ax1.annotate(f"lowest latency at {xs[best]:.0f} ms,\nbut ~{spur[best]:.0f} wasted retx/100 req",
                 (xs[best], lat[best]),
                 xytext=(xs[best] * 1.3, lat[best] + 60), color=TEXT2,
                 arrowprops=dict(arrowstyle="-", color=TEXT2, lw=0.8))
    legend_below(ax1)
    ax2.plot(xs, spur, color=cF, marker="o", ms=5, label="Fixed timeout")
    ax2.axhline(a["spurious"] / 9, color=cA, ls="--", label="Adaptive (Jacobson/Karels)")
    ax2.set_xscale("log")
    ax2.set_xlabel("Fixed timeout value (ms, log scale)")
    ax2.set_ylabel("Spurious retx per 100 requests")
    ax2.set_title("(b) Wasted retransmissions", loc="left")
    legend_below(ax2)
    fig.suptitle("Fig 4. The fixed-timeout dilemma: too short wastes packets, "
                 "too long wastes time", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPH_DIR, "fig4_fixed_timeout_tradeoff.png"))
    plt.close(fig)


def exp4_jitter_sweep():
    """Graph 5: why the variance term (RTTVAR) matters."""
    print("Experiment 4: jitter sweep (EWMA vs Jacobson)")
    jitters = [0.005, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
    rows = []
    out = {n: [] for n in ("Adaptive (Jacobson/Karels)", "EWMA (RFC 793)")}
    for j in jitters:
        for name in out:
            s = averaged(lambda sd: Network([(0, 0.150, j)], loss=0.02,
                                            spike_p=0.0, seed=sd),
                         lambda: strategies()[name], 500)
            out[name].append(s)
            rows.append({"jitter_ms": j * 1000, "strategy": name,
                         **{k: round(v, 4) for k, v in s.items()}})
    save_csv("exp4_jitter_sweep.csv", rows)
    xs = [1000 * j for j in jitters]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for name, ss in out.items():
        c, ls = STYLE[name]
        ax1.plot(xs, [s["spurious"] / 5 for s in ss], color=c, ls=ls,
                 marker="o", ms=5, label=name)
        ax2.plot(xs, [s["p95_latency_ms"] for s in ss], color=c, ls=ls,
                 marker="o", ms=5, label=name)
    for ax in (ax1, ax2):
        ax.set_xlabel("RTT jitter, std-dev (ms)   [mean RTT = 150 ms]")
        legend_below(ax)
    ax1.set_ylabel("Spurious retx per 100 requests")
    ax1.set_title("(a) False timeouts", loc="left")
    ax2.set_ylabel("95th-percentile latency (ms)")
    ax2.set_title("(b) Tail latency", loc="left")
    fig.suptitle("Fig 5. Tracking RTT variance (RTTVAR) keeps the timeout safe "
                 "as jitter grows", x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPH_DIR, "fig5_jitter_sensitivity.png"))
    plt.close(fig)


def exp5_outage_backoff():
    """Graph 6: exponential back-off during a 10-second link outage."""
    print("Experiment 5: outage & exponential back-off")
    outage = (5.0, 15.0)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    rows = []
    variants = [("Adaptive + exponential back-off", JacobsonKarels(),
                 STYLE["Adaptive (Jacobson/Karels)"][0], "-"),
                ("Adaptive, no back-off", JacobsonNoBackoff(),
                 STYLE["EWMA (RFC 793)"][0], "--")]
    for label, est, c, ls in variants:
        net = Network([(0, 0.100, 0.010)], loss=0.0, spike_p=0.0,
                      outage=outage, seed=3)
        net.send_log = []          # record every transmission time
        sends = net.send_log
        recs, s = simulate(est, net, 200, max_tx=30)
        in_outage = sum(outage[0] <= t < outage[1] for t in sends)
        rows.append({"variant": label, "packets_sent_during_outage": in_outage,
                     **{k: round(v, 4) for k, v in s.items()}})
        # cumulative transmissions over time
        sends.sort()
        ax.step(sends, np.arange(1, len(sends) + 1), where="post", color=c, ls=ls,
                label=f"{label}  ({in_outage} packets into the dead link)")
    ax.axvspan(*outage, color=GRID, alpha=0.6, lw=0)
    ax.text(sum(outage) / 2, ax.get_ylim()[1] * 0.92, "link outage (10 s)",
            ha="center", color=TEXT2)
    ax.set_xlim(0, 30)
    ax.set_xlabel("Simulated time (s)")
    ax.set_ylabel("Cumulative packets transmitted")
    ax.set_title("Fig 6. Exponential back-off stops flooding a dead link",
                 loc="left")
    legend_below(ax, ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPH_DIR, "fig6_outage_backoff.png"))
    plt.close(fig)
    save_csv("exp5_outage.csv", rows)


def exp6_summary():
    """Graph 7: overall scorecard on the 3-phase scenario with 5 % loss."""
    print("Experiment 6: overall comparison")
    res = {}
    rows = []
    for name in STYLE:
        res[name] = averaged(lambda sd: Network(PHASES, loss=0.05, seed=sd),
                             lambda: strategies()[name], 900, seeds=range(20))
        rows.append({"strategy": name, **{k: round(v, 4) for k, v in res[name].items()}})
    save_csv("exp6_summary.csv", rows)

    metrics = [("mean_latency_ms", "Mean latency (ms)", 1, "{:.0f}"),
               ("p95_latency_ms", "95th-percentile latency (ms)", 1, "{:.0f}"),
               ("spurious", "Spurious retx / 100 req", 1 / 9, "{:.1f}"),
               ("efficiency", "Transmission efficiency (%)", 100, "{:.1f}")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.4))
    names = list(STYLE)
    short = ["Adaptive", "EWMA", "Fixed\n200 ms", "Fixed\n1 s"]
    for ax, (key, title, scale, fmt) in zip(axes, metrics):
        vals = [res[n][key] * scale for n in names]
        bars = ax.bar(short, vals, color=[STYLE[n][0] for n in names],
                      width=0.62, edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v),
                    ha="center", va="bottom", color=TEXT1, fontsize=9)
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.15)
        if key == "efficiency":
            ax.set_ylim(min(vals) - 10, 100)
    fig.suptitle("Fig 7. Overall comparison (3-phase network, 5 % loss, 20 runs)",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPH_DIR, "fig7_summary.png"))
    plt.close(fig)
    return res


def exp_rtt_robustness():
    """Graph 8: one timeout policy, many networks (LAN -> satellite)."""
    print("Experiment 7: robustness across network RTTs")
    nets = [("LAN", 0.005), ("Metro", 0.030), ("National", 0.080),
            ("Intercontinental", 0.200), ("Mobile 3G", 0.400),
            ("Satellite (GEO)", 0.600)]
    rows = []
    res = {n: [] for n in STYLE}
    for label, base in nets:
        for name in STYLE:
            s = averaged(lambda sd: Network([(0, base, 0.15 * base)], loss=0.05,
                                            spike_p=0.0, min_rtt=0.001, seed=sd),
                         lambda: strategies()[name], 400)
            s["overhead"] = s["mean_latency_ms"] / (1000 * base)
            res[name].append(s)
            rows.append({"network": label, "base_rtt_ms": base * 1000,
                         "strategy": name, **{k: round(v, 4) for k, v in s.items()}})
    save_csv("exp7_rtt_robustness.csv", rows)
    xs = np.arange(len(nets))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for name, (c, ls) in STYLE.items():
        ax1.plot(xs, [s["overhead"] for s in res[name]], color=c, ls=ls,
                 marker="o", ms=5, label=name)
        ax2.plot(xs, [100 * s["efficiency"] for s in res[name]], color=c, ls=ls,
                 marker="o", ms=5, label=name)
    ticks = [f"{l}\n{b*1000:.0f} ms" for l, b in nets]
    for ax in (ax1, ax2):
        ax.set_xticks(xs, ticks, fontsize=8.5)
        legend_below(ax)
    ax1.set_yscale("log")
    ax1.set_ylabel("Mean latency / base RTT  (1 = ideal, log)")
    ax1.set_title("(a) Latency overhead", loc="left")
    ax2.set_ylabel("Transmission efficiency (%)")
    ax2.set_title("(b) Packets wasted", loc="left")
    fig.suptitle("Fig 8. A fixed timeout is tuned for one network; the adaptive "
                 "RTO works on all of them (5 % loss)", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPH_DIR, "fig8_rtt_robustness.png"))
    plt.close(fig)


if __name__ == "__main__":
    exp1_rto_tracking()
    exp2_loss_sweep()
    exp3_fixed_timeout_tradeoff()
    exp4_jitter_sweep()
    exp5_outage_backoff()
    res = exp6_summary()
    exp_rtt_robustness()
    print("\n=== Overall comparison (Fig 7) ===")
    hdr = f"{'strategy':28s} {'mean ms':>8s} {'p95 ms':>8s} {'spur/100':>9s} {'eff %':>6s}"
    print(hdr)
    for n, s in res.items():
        print(f"{n:28s} {s['mean_latency_ms']:8.1f} {s['p95_latency_ms']:8.1f} "
              f"{s['spurious'] / 9:9.2f} {100 * s['efficiency']:6.1f}")
    print(f"\nGraphs written to {GRAPH_DIR}")
