"""
sim_core.py
-----------
Pure-Python simulation engine (no numpy / matplotlib needed).
Used by simulation.py (which draws the report graphs) and by app.py (the UI).

Network model (per transmission):
    * lost with probability p (request or reply)
    * otherwise RTT ~ max(min_rtt, Normal(base_rtt, jitter)) plus occasional
      delay spikes (queue build-ups)
    * an optional outage window during which every packet is lost

A late ACK of an earlier transmission still completes the request (the
client accepts the first matching ACK) - so a timer that fires while the
original ACK is still in flight produces a *spurious* retransmission.
"""
import random
from statistics import mean

from rto_estimator import FixedTimeout, EWMATimeout, JacobsonKarels

NAN = float("nan")

# The four strategies compared throughout the project
STRATEGY_NAMES = ["Adaptive (Jacobson/Karels)", "EWMA (RFC 793)",
                  "Fixed 200 ms", "Fixed 1 s"]


def strategies(fixed_short=0.2, fixed_long=1.0):
    """Fresh estimator instances (name -> estimator)."""
    return {
        "Adaptive (Jacobson/Karels)": JacobsonKarels(),
        "EWMA (RFC 793)": EWMATimeout(),
        "Fixed 200 ms": FixedTimeout(fixed_short),
        "Fixed 1 s": FixedTimeout(fixed_long),
    }


def make_strategy(name, fixed_short=0.2, fixed_long=1.0):
    return strategies(fixed_short, fixed_long)[name]


class JacobsonNoBackoff(JacobsonKarels):
    """Same estimator with exponential back-off disabled (for the outage test)."""
    name = "Adaptive, no back-off"

    def on_timeout(self):
        pass


def percentile(values, q):
    """Linear-interpolated percentile, same result as numpy.percentile."""
    v = sorted(values)
    if not v:
        return NAN
    k = (len(v) - 1) * q / 100.0
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


class Network:
    """
    phases: list of (start_request_index, base_rtt_s, jitter_s)
    loss:   probability that a single transmission (or its ACK) is lost
    spike:  probability that a transmission sees an extra queueing spike
    outage: (start_time_s, end_time_s) during which every packet is lost
    """
    def __init__(self, phases, loss=0.05, spike_p=0.02, spike_ms=(150, 400),
                 min_rtt=0.005, outage=None, seed=0):
        self.phases = sorted(phases)
        self.loss = loss
        self.spike_p, self.spike_ms = spike_p, spike_ms
        self.min_rtt = min_rtt
        self.outage = outage
        self.rng = random.Random(seed)
        self.send_log = None      # set to [] to record every send time

    def params(self, req_idx):
        base, jit = self.phases[0][1], self.phases[0][2]
        for start, b, j in self.phases:
            if req_idx >= start:
                base, jit = b, j
        return base, jit

    def transmit(self, req_idx, now):
        """Return the RTT of this transmission, or None if it is lost."""
        if self.send_log is not None:
            self.send_log.append(now)
        if self.outage and self.outage[0] <= now < self.outage[1]:
            return None
        if self.rng.random() < self.loss:
            return None
        base, jit = self.params(req_idx)
        rtt = max(self.min_rtt, self.rng.gauss(base, jit))
        if self.rng.random() < self.spike_p:
            rtt += self.rng.uniform(*self.spike_ms) / 1000
        return rtt


def simulate(est, net, n_requests, max_tx=12, think_time=0.0):
    """Stop-and-wait ARQ.  Returns per-request records and a summary dict."""
    now = 0.0
    records = []
    tx_total = spurious = failures = 0
    for i in range(n_requests):
        start = now
        first_rto = est.rto()
        arrivals = []          # absolute ACK arrival times of all copies
        attempts = 0
        done_at = None
        while attempts < max_tx:
            rto = est.rto()
            attempts += 1
            tx_total += 1
            rtt = net.transmit(i, now)
            if rtt is not None:
                arrivals.append(now + rtt)
            deadline = now + rto
            first = min(arrivals) if arrivals else None
            if first is not None and first <= deadline:
                done_at = first
                if attempts == 1:                 # Karn's algorithm
                    est.on_rtt_sample(first - start)
                break
            if first is not None:                 # ACK still in flight
                spurious += 1
            est.on_timeout()
            now = deadline
        if done_at is None:
            failures += 1
            done_at = now
        latency = done_at - start
        records.append({
            "req": i, "time_s": start, "attempts": attempts,
            "rtt_s": (done_at - start) if attempts == 1 and arrivals else NAN,
            "true_base_rtt_s": net.params(i)[0],
            "rto_s": first_rto, "latency_s": latency,
        })
        now = done_at + think_time
    n = n_requests
    lats = [r["latency_s"] for r in records]
    summary = {
        "transmissions": tx_total,
        "retransmissions": tx_total - n,
        "spurious": spurious,
        "failures": failures,
        "mean_latency_ms": 1000 * mean(lats),
        "p95_latency_ms": 1000 * percentile(lats, 95),
        "efficiency": n / tx_total,                    # useful / total packets
        "throughput_rps": n / now if now > 0 else 0,   # requests per second
        "total_time_s": now,
    }
    return records, summary


def averaged(make_net, make_est, n_requests, seeds=range(10)):
    """Run several seeds and average the summary metrics."""
    sums = [simulate(make_est(), make_net(s), n_requests)[1] for s in seeds]
    return {k: mean(s[k] for s in sums) for k in sums[0]}


# The 3-phase scenario used in Figs 1, 4 and 7
PHASES = [(0, 0.100, 0.010),      # normal: 100 ms +/- 10
          (300, 0.300, 0.060),    # congestion: 300 ms +/- 60
          (600, 0.080, 0.008)]    # recovered fast path: 80 ms +/- 8
