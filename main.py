#!/usr/bin/env python3
"""polymarket-nvidia-ceo-forecast-01: a random-walk persistence forecast.

For each case file ``<id>.in`` in $MAKEFUN_INPUT_DIR (default /input), read in name order, one
market's Yes price at the checkpoint is ``x`` and its spread is the population standard deviation
(divide by n) of the last 240 one-minute differences, scaled by 0.375 per square-root minute:

    q_tau(h) = clamp(x + z_tau * max(1e-5, 0.375 * sigma1 * sqrt(h)), 0, 1)

rounded to 6 decimals, for h = 1..horizon_minutes. The five levels are 0.05, 0.25, 0.5, 0.75,
0.95 in that order, and each row is forced non-decreasing. Writes ``<id>.out`` as compact JSON
``{"forecasts":[{"id":...,"q":[[...],...]}]}`` and nothing else. Standard library only, offline,
deterministic: the same case always yields the same bytes.
"""

import json
import math
import os
import sys

# The standard normal's quantile at each level, in the row order 0.05, 0.25, 0.5, 0.75, 0.95.
Z = (-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722)
# Spread of one minute's move, in price units per square-root minute, per unit of sigma1.
SPREAD = 0.375
# Fewer than two prices means no differences to measure: the baseline's 0.0025 stands in.
FALLBACK_SIGMA = 0.0025
# The spread floor, so a flat history still yields a strictly positive interval.
SIGMA_FLOOR = 1e-5
# Differences read from the history (so up to 241 prices).
DIFF_WINDOW = 240
# Micro-units per price unit: 6 decimals exactly.
MICRO = 1000000


def sigma1(prices):
    """Population standard deviation (divide by n) of the last 240 one-minute differences."""
    if len(prices) < 2:
        return FALLBACK_SIGMA
    window = prices[-(DIFF_WINDOW + 1):]
    diffs = [window[i] - window[i - 1] for i in range(1, len(window))]
    n = len(diffs)
    mean = math.fsum(diffs) / n
    var = math.fsum((d - mean) ** 2 for d in diffs) / n
    return math.sqrt(var) if var > 0.0 else 0.0


def clean_prices(raw):
    """The finite numbers of a market's price list, clamped into [0,1], in order."""
    out = []
    for v in raw:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(0.0 if f <= 0.0 else (1.0 if f >= 1.0 else f))
    return out


def quantize(x):
    """x clamped to [0,1] as integer micro-units: round half up, so never -0.0."""
    if not math.isfinite(x) or x <= 0.0:
        return 0
    if x >= 1.0:
        return MICRO
    us = math.floor(x * MICRO + 0.5)
    return MICRO if us > MICRO else us


def fmt(us):
    """One integer micro-unit count as plain decimal text: at most 6 places, no exponent."""
    whole, frac = divmod(us, MICRO)
    return "%d.%06d" % (whole, frac)


def forecast(prices, horizon):
    """The horizon rows of the five quantiles for one market, as JSON number text."""
    prices = clean_prices(prices) if prices else []
    x = prices[-1] if prices else 0.5
    if not math.isfinite(x):
        x = 0.5
    x = min(1.0, max(0.0, x))
    s1 = sigma1(prices)
    out = []
    for h in range(1, horizon + 1):
        sigma = SPREAD * s1 * math.sqrt(h)
        if sigma < SIGMA_FLOOR:
            sigma = SIGMA_FLOOR
        row = []
        prev = 0
        for z in Z:
            us = quantize(x + z * sigma)
            if us < prev:
                us = prev
            prev = us
            row.append(fmt(us))
        out.append("[" + ",".join(row) + "]")
    return out


def read_case(path):
    """One case file as (horizon, markets); a market is a dict with a string id and prices."""
    with open(path, "r", encoding="utf-8") as fh:
        case = json.load(fh)
    if not isinstance(case, dict):
        raise ValueError("case is not a JSON object")
    try:
        horizon = int(case.get("horizon_minutes") or 240)
    except (TypeError, ValueError):
        horizon = 240
    if horizon < 1:
        horizon = 240
    markets = []
    for m in case.get("markets") or []:
        if not isinstance(m, dict) or m.get("id") is None:
            continue
        prices = m.get("prices")
        markets.append((str(m["id"]), prices if isinstance(prices, list) else []))
    return horizon, markets


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        sys.stderr.write("cannot create %s: %s\n" % (out_dir, exc))
        return 2

    try:
        names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    except OSError as exc:
        # No case to read (or no directory): nothing to write, and that is not a crash.
        sys.stderr.write("cannot list %s: %s\n" % (in_dir, exc))
        return 0

    for name in names:
        path = os.path.join(in_dir, name)
        if not os.path.isfile(path):
            continue
        stem = name[: -len(".in")]
        try:
            horizon, markets = read_case(path)
            parts = []
            for mid, prices in markets:
                rows = forecast(prices, horizon)
                parts.append('{"id":%s,"q":[%s]}' % (json.dumps(mid), ",".join(rows)))
            body = '{"forecasts":[%s]}\n' % ",".join(parts)
            with open(os.path.join(out_dir, stem + ".out"), "w", encoding="utf-8") as fh:
                fh.write(body)
        except Exception as exc:  # one unreadable case keeps the others gradeable
            sys.stderr.write("skipping %s: %s\n" % (name, exc))

    return 0


if __name__ == "__main__":
    sys.exit(main())
