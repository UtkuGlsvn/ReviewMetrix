"""IP-based rate limiting for the scraping routes.

Scraping is the scarce resource here: the stores rate-limit and eventually
block callers, so an open deployment needs a ceiling. The limiter counts
*scrape cost* rather than requests, because a six-country comparison performs
six scrapes and should not cost the same as a single analysis.

Deliberately in-memory: a single free-tier instance needs no database, and a
counter check costs nothing. Running more than one process gives each its own
counter, so the effective limit multiplies by the process count — see
README for when a shared store becomes necessary.
"""
import os
import time
import threading
from collections import defaultdict, deque

# Scrape units allowed per IP per window
RATE_LIMIT_MAX = int(os.environ.get('RATE_LIMIT_MAX', 30))
RATE_LIMIT_WINDOW = int(os.environ.get('RATE_LIMIT_WINDOW', 3600))  # seconds

# Guards against unbounded growth from many distinct IPs
MAX_TRACKED_KEYS = 5000


class RateLimiter:
    """Sliding-window limiter keyed by client IP."""

    def __init__(self, limit=RATE_LIMIT_MAX, window=RATE_LIMIT_WINDOW):
        self.limit = limit
        self.window = window
        self._hits = defaultdict(deque)   # key -> deque of (timestamp, cost)
        self._lock = threading.Lock()

    def _prune(self, key, now):
        """Drop entries that fell out of the window. Caller must hold the lock."""
        dq = self._hits[key]
        cutoff = now - self.window
        while dq and dq[0][0] <= cutoff:
            dq.popleft()
        return dq

    def _evict_stale(self, now):
        """Forget IPs with no activity in the window. Caller must hold the lock."""
        cutoff = now - self.window
        for key in [k for k, dq in self._hits.items() if not dq or dq[-1][0] <= cutoff]:
            del self._hits[key]

    def check(self, key, cost=1):
        """Record `cost` units against `key`.

        Returns (allowed, remaining, retry_after_seconds). Nothing is recorded
        when the request is rejected, so a blocked caller cannot push their own
        reset further away by retrying.
        """
        now = time.time()
        with self._lock:
            if len(self._hits) > MAX_TRACKED_KEYS:
                self._evict_stale(now)

            dq = self._prune(key, now)
            used = sum(c for _, c in dq)

            if used + cost > self.limit:
                retry = int(dq[0][0] + self.window - now) + 1 if dq else self.window
                return False, max(0, self.limit - used), max(1, retry)

            dq.append((now, cost))
            return True, self.limit - used - cost, 0

    def reset(self):
        """Clear all counters (used by tests)."""
        with self._lock:
            self._hits.clear()

    def usage(self, key):
        """Units currently counted against a key."""
        with self._lock:
            return sum(c for _, c in self._prune(key, time.time()))


limiter = RateLimiter()


def client_ip(request):
    """Best-effort client IP.

    X-Forwarded-For is only honoured when TRUST_PROXY=1, because any client can
    set that header directly. Enable it when running behind a proxy that
    overwrites the header (Render, Fly, a reverse proxy you control); leaving it
    off would otherwise let anyone bypass the limit by forging a new IP.
    """
    if os.environ.get('TRUST_PROXY') == '1':
        forwarded = request.headers.get('X-Forwarded-For', '')
        if forwarded:
            return forwarded.split(',')[0].strip()
    return request.remote_addr or 'unknown'
