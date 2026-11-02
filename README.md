# Adaptive Timeout Optimization for Reliable Client-Server Communication

Network Programming mini project. A reliable request/response protocol over UDP
whose retransmission timeout (RTO) adapts to measured round-trip time, compared
against fixed and mean-only timers, with an interactive web dashboard.

## Quick start: double-click `AdaptiveTimeoutLab.html`

It opens in your browser. No Terminal, no Python, no installs, and it works offline.

| View | What it shows |
|---|---|
| Live demo | Four clients (Adaptive, EWMA, short fixed, long fixed) send requests over the same simulated network. Every request, ACK, loss, timeout and retransmission is **animated as it happens**, with a live event log, scorecard, RTO-vs-RTT graphs and packets-over-time graph. Inputs: scenario (steady / congestion / outage), base RTT, jitter, packet loss, requests, fixed timeouts, random seed. Speed 0.1× to 25×, Pause (Space), Restart, Skip to end. |
| Experiments | Interactive sweeps: LAN to satellite, fixed-timeout dilemma, jitter, loss, outage and back-off |

The page's source is in `web/` (`engine.js` is a JavaScript port of
`rto_estimator.py` + `sim_core.py`). Rebuild it with `python3 web/build.py`.

### Optional: the older Python dashboard with real UDP sockets

Browsers cannot open raw UDP sockets, so the real-socket test needs Python:
`python3 app.py` (its "Live sockets" tab starts real UDP servers and clients),
or `python3 live_experiment.py` for Figs 9–10.

## Files

| File | What it does |
|---|---|
| `AdaptiveTimeoutLab.html` | **The dashboard: double-click to open** |
| `web/` | Source of the dashboard (engine, charts, UI) and its build script |
| `app.py` | Older Python dashboard with real UDP sockets (optional) |
| `static/` | Dashboard page, styles, and a small SVG chart engine |
| `rto_estimator.py` | Timeout strategies: `FixedTimeout`, `EWMATimeout` (RFC 793), `JacobsonKarels` (RFC 6298 + Karn + exponential back-off) |
| `server.py` | UDP server that emulates loss, delay, jitter and congestion |
| `client.py` | Reliable stop-and-wait UDP client using any estimator; logs to CSV |
| `sim_core.py` | Discrete-event simulation engine (pure Python) |
| `simulation.py` | Runs 7 experiments and draws `graphs/fig1..fig8` (needs numpy + matplotlib) |
| `live_experiment.py` | Runs the real server + client per strategy, draws `graphs/fig9..fig10` |
| `graphs/` | Generated figures (PNG) |
| `results/` | Raw numbers behind every figure (CSV) |

## Command-line versions

    # Regenerate the report graphs (needs: pip install numpy matplotlib)
    python3 simulation.py          # Figs 1-8, about 5 seconds
    python3 live_experiment.py     # Figs 9-10, about 4 minutes

    # Manual client/server demo in two terminals
    python3 server.py --loss 0.1 --delay 0.05 --jitter 0.02 --congest-after 100 --congest-delay 0.25
    python3 client.py --mode adaptive -n 200
    python3 client.py --mode fixed --fixed-timeout 0.1 -n 200

## Adaptive RTO (RFC 6298)

    first sample:  SRTT = R,  RTTVAR = R/2
    afterwards:    RTTVAR = (1 - 1/4) * RTTVAR + 1/4 * |SRTT - R|
                   SRTT   = (1 - 1/8) * SRTT   + 1/8 * R
    RTO = SRTT + max(G, 4 * RTTVAR)
    on timeout:    RTO = RTO * 2   (exponential back-off)
    Karn's rule:   never take an RTT sample from a retransmitted packet
