"""
rto_estimator.py
----------------
Retransmission Timeout (RTO) estimators used by both the real UDP
client and the simulation.

Implemented strategies
  1. FixedTimeout         - constant timeout (the naive baseline)
  2. EWMATimeout          - original RFC 793 style: SRTT smoothed average, RTO = beta * SRTT
  3. JacobsonKarels       - RFC 6298: SRTT + 4 * RTTVAR, with Karn's algorithm
                            and exponential back-off (the "adaptive" scheme)

Every estimator exposes the same interface:
    rto()                 -> current timeout in seconds
    on_rtt_sample(rtt)    -> feed a valid RTT measurement
    on_timeout()          -> called when a timer expires (back-off)
"""


class FixedTimeout:
    name = "Fixed"

    def __init__(self, timeout=1.0):
        self.timeout = timeout

    def rto(self):
        return self.timeout

    def on_rtt_sample(self, rtt):
        pass  # never adapts

    def on_timeout(self):
        pass  # never backs off


class EWMATimeout:
    """RFC 793 (1981): SRTT = a*SRTT + (1-a)*RTT ; RTO = beta*SRTT"""
    name = "EWMA (RFC 793)"

    def __init__(self, alpha=0.875, beta=2.0, initial=1.0,
                 min_rto=0.05, max_rto=60.0):
        self.alpha, self.beta = alpha, beta
        self.srtt = None
        self.initial = initial
        self.min_rto, self.max_rto = min_rto, max_rto
        self.backoff = 1

    def rto(self):
        base = self.initial if self.srtt is None else self.beta * self.srtt
        return min(self.max_rto, max(self.min_rto, base * self.backoff))

    def on_rtt_sample(self, rtt):
        self.srtt = rtt if self.srtt is None else \
            self.alpha * self.srtt + (1 - self.alpha) * rtt
        self.backoff = 1

    def on_timeout(self):
        self.backoff = min(self.backoff * 2, 64)


class JacobsonKarels:
    """
    RFC 6298 (Jacobson/Karels 1988):
        first sample:  SRTT = R, RTTVAR = R/2
        afterwards:    RTTVAR = (1-b)*RTTVAR + b*|SRTT - R|
                       SRTT   = (1-a)*SRTT   + a*R
        RTO = SRTT + max(G, K*RTTVAR)     a=1/8, b=1/4, K=4
    Plus:
        Karn's algorithm   - RTT of retransmitted packets is ignored
                             (enforced by the caller: only pass
                             samples from packets sent once)
        Exponential backoff - RTO doubles on every timeout
    """
    name = "Adaptive (Jacobson/Karels)"

    def __init__(self, alpha=1/8, beta=1/4, k=4, initial=1.0,
                 min_rto=0.05, max_rto=60.0, granularity=0.001):
        self.alpha, self.beta, self.k = alpha, beta, k
        self.srtt = None
        self.rttvar = None
        self.initial = initial
        self.min_rto, self.max_rto = min_rto, max_rto
        self.g = granularity
        self.backoff = 1

    def _base(self):
        if self.srtt is None:
            return self.initial
        return self.srtt + max(self.g, self.k * self.rttvar)

    def rto(self):
        return min(self.max_rto, max(self.min_rto, self._base() * self.backoff))

    def on_rtt_sample(self, rtt):
        if self.srtt is None:
            self.srtt = rtt
            self.rttvar = rtt / 2
        else:
            self.rttvar = (1 - self.beta) * self.rttvar + \
                self.beta * abs(self.srtt - rtt)
            self.srtt = (1 - self.alpha) * self.srtt + self.alpha * rtt
        self.backoff = 1  # a fresh valid sample collapses the back-off

    def on_timeout(self):
        self.backoff = min(self.backoff * 2, 64)


def make_estimator(kind, **kw):
    kind = kind.lower()
    if kind == "fixed":
        return FixedTimeout(**kw)
    if kind == "ewma":
        return EWMATimeout(**kw)
    if kind in ("adaptive", "jacobson", "jk"):
        return JacobsonKarels(**kw)
    raise ValueError(f"unknown estimator: {kind}")
