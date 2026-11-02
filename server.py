"""
server.py
---------
UDP request/response server that EMULATES an unreliable network:
  * random packet loss (request silently dropped)
  * variable delay (base delay + random jitter)
  * optional "congestion" mode where delay jumps up after N packets

Each reply is sent from its own timer thread so a slow reply does not
block other requests (just like a real router queue).

Usage (command line):
    python server.py --port 9999 --loss 0.1 --delay 0.05 --jitter 0.02
    python server.py --congest-after 100 --congest-delay 0.25

Usage (from Python, e.g. the UI):
    stop = threading.Event()
    threading.Thread(target=serve, kwargs=dict(port=9999, stop_event=stop)).start()
    ...
    stop.set()
"""
import argparse
import random
import socket
import threading
import time


def serve(host="127.0.0.1", port=9999, loss=0.10, delay=0.05, jitter=0.02,
          congest_after=0, congest_delay=0.25, stop_event=None, verbose=True,
          ready_event=None, info=None):
    """Run the lossy server. port=0 picks a free port; it is written to info['port'].
    info['packets'] counts packets received (used by the UI)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.settimeout(0.2)  # wake up regularly to check stop_event
    if info is not None:
        info["port"] = sock.getsockname()[1]
        info["packets"] = 0
        info["dropped"] = 0
    if verbose:
        print(f"[server] listening on {host}:{port}  loss={loss} "
              f"delay={delay}s jitter={jitter}s")
    if ready_event is not None:
        ready_event.set()

    count = 0
    lock = threading.Lock()

    def delayed_reply(data, addr, d):
        time.sleep(d)
        try:
            sock.sendto(b"ACK:" + data, addr)
        except OSError:
            pass  # socket closed while reply was "in flight"

    try:
        while stop_event is None or not stop_event.is_set():
            try:
                data, addr = sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            if data == b"SHUTDOWN":
                if verbose:
                    print("[server] shutdown requested")
                break
            with lock:
                count += 1
                n = count
            if info is not None:
                info["packets"] = n
            if random.random() < loss:
                if info is not None:
                    info["dropped"] += 1
                continue  # emulate a lost packet
            base = congest_delay if (congest_after and n > congest_after) else delay
            d = max(0.0, random.gauss(base, jitter))
            threading.Thread(target=delayed_reply, args=(data, addr, d),
                             daemon=True).start()
    except KeyboardInterrupt:
        if verbose:
            print("\n[server] stopped")
    finally:
        time.sleep(0.05)
        sock.close()


def main():
    ap = argparse.ArgumentParser(description="Lossy UDP echo server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--loss", type=float, default=0.10, help="drop probability 0..1")
    ap.add_argument("--delay", type=float, default=0.05, help="base delay (s)")
    ap.add_argument("--jitter", type=float, default=0.02, help="std-dev of delay (s)")
    ap.add_argument("--congest-after", type=int, default=0,
                    help="after this many packets, switch to congestion delay (0=off)")
    ap.add_argument("--congest-delay", type=float, default=0.25)
    a = ap.parse_args()
    serve(a.host, a.port, a.loss, a.delay, a.jitter, a.congest_after, a.congest_delay)


if __name__ == "__main__":
    main()
