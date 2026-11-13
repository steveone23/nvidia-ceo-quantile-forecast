"""Quantile forecaster for the make.fun bounty polymarket-nvidia-ceo-forecast-01.

Mature markets: persistence median with a quantile fan whose width at each
horizon is a non-negative linear blend of trailing realised absolute price
moves of the same market. Young markets (less than about 25 hours of prices)
get their own fitted fan and a shifted median, because their prices are still
decaying from wide opening quotes. All weights were fitted offline by linear
quantile regression (pinball loss) and are constants here.

Standard library only, deterministic, no network, no clock.
"""

import json
import math
import os
import sys

GRID = [1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 60, 75, 90, 105, 120, 150, 180, 210, 240]
WINDOWS = [120, 360, 1440]          # minutes, for the mean absolute h-minute move
HALF_LIVES = [15.0, 60.0, 240.0]    # minutes, for the EWMA of absolute 1-minute moves
EWMA_SPAN = 1440                    # only the last 1440 one-minute moves enter the EWMAs
LOOKBACK = 1680                     # minutes of history used
FULL_HISTORY = 4320
OPENING_SKIP = 360                  # a market's first 6 hours are a wild opening auction, not a guide
MIN_KEEP = 120
MIN_POINTS = 62
COEF_75_MEAN = [0.043097160041674555, 0.0, 0.04681729364445213]
COEF_75_EWMA = [0.22386859805118922, 0.22640690896668184, 0.0]
COEF_95_MEAN = [0.24599069369676957, 0.33398800825252445, 0.5581926205918781]
COEF_95_EWMA = [1.08176170092967, 0.017106602489334325, 0.0]
FALLBACK_SIGMA = 0.0025
Z75 = 0.67449
Z95 = 1.644854

# Young markets: fewer than YOUNG_N prices in the case (under 25 hours of trading).
YOUNG_N = 1500
NEWBORN_K75U, NEWBORN_K95U, NEWBORN_K75D, NEWBORN_K95D, NEWBORN_BETA = (0.05, 5.5, 1.6, 2.5, 0.5)
YOUNG_GROUPS = [
    {
        "max_n": 500,
        "gamma": [2.2015273860041407, -1.836478367926592, 3.531688532278698, -1.5271556978238978, -0.31639214818386935, 38.54124765698144, -31.00977697629384, -4.2204717208831335, 1.8264497424650699, 0.34444580482469966],
        "u75": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1255744406557293, 1.707319917261436],
        "u95": [0.0, 0.40685319902466666, 0.0, 0.0, 0.0, 0.0, 0.1270847240143684, 6.335014801743384],
        "d25": [0.0, 0.0, 0.0, 0.0, 0.0, 0.2935822930499664, 0.0, 0.18673184525543932],
        "d05": [0.0, 0.0, 0.0, 0.0, 0.0, 0.4474978101333309, 0.5792505804740685, 3.763203844633466],
    },
    {
        "max_n": 900,
        "gamma": [-0.10216724918114337, 0.12470349889944098, 0.20906856387944447, 0.6462745424983789, -0.8010728464651685, 0.18176183479508606, 1.2926107139367253, 0.6365020885471466, -1.5594075626895691, 1.8133104929794297],
        "u75": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6226775463418164, 1.7077205946643703],
        "u95": [0.0, 0.0, 0.6945241266297069, 0.39507465701160954, 0.0, 0.04356784699045317, 0.0, 12.41551545899898],
        "d25": [0.027493515219369092, 0.0, 0.0, 0.0, 0.0, 0.23942869847009285, 0.09696276679208966, 0.32509793368094936],
        "d05": [0.0, 0.2604774144029427, 0.16228857435890146, 0.0, 0.0, 0.0, 1.4587732511765221, 2.2016743102648806],
    },
    {
        "max_n": 1500,
        "gamma": [-0.5507040533778228, 0.5305720531042784, 3.3909907685774527, 0.18479679623438222, -0.5131424201191751, -10.01903401135634, 7.8005815915868695, -0.33198683822772146, -2.845006950151454, 1.271984655075666],
        "u75": [0.0, 0.0, 0.0, 0.12611426494368141, 0.0, 0.0, 0.07442602084289036, 0.05092845954844046],
        "u95": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16845951400763876, 4.212762010143697],
        "d25": [0.0, 0.0, 0.07962698060952611, 0.0, 0.0, 0.1403731666440754, 0.0, 0.7066916805118978],
        "d05": [0.0, 0.0, 0.04553972860126844, 0.0, 0.0, 0.0645369923594319, 0.5420091041472519, 3.0832545871611474],
    },
]

DEFAULT_HORIZON = 240


def sanitise(raw):
    """Forward-fill non-finite values, drop leading invalid ones, clamp to [0, 1]."""
    out = []
    if not isinstance(raw, list):
        return out
    prev = None
    for v in raw:
        ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        if ok:
            prev = min(1.0, max(0.0, float(v)))
            out.append(prev)
        elif prev is not None:
            out.append(prev)
    return out


def usable_history(p):
    """Drop a market's opening when the case holds its whole life, then keep LOOKBACK points."""
    n = len(p)
    if n < FULL_HISTORY:
        s = min(OPENING_SKIP, max(0, n - MIN_KEEP))
        u = p[s:]
    else:
        u = p
    return u[-LOOKBACK:]


def volatility_features(u):
    """EWMAs of absolute 1-minute moves and grid window means of absolute h-minute moves.

    Requires len(u) >= MIN_POINTS. Returns (ew, means) with ew[li] per half-life
    and means[wi][j] per window and grid horizon.
    """
    m = len(u)
    d1 = [abs(u[k + 1] - u[k]) for k in range(m - 1)]
    d1 = d1[-EWMA_SPAN:]
    nd = len(d1)
    ew = []
    for half_life in HALF_LIVES:
        num = 0.0
        den = 0.0
        for k in range(nd):
            w = 0.5 ** ((nd - 1 - k) / half_life)
            num += d1[k] * w
            den += w
        ew.append(num / den)

    means = []
    for window in WINDOWS:
        row = []
        last_valid = -1
        for j, h in enumerate(GRID):
            wp = max(window, 3 * h)
            e0 = max(h, m - wp)
            cnt = m - e0
            if cnt >= max(30, h // 2):
                total = 0.0
                for e in range(e0, m):
                    total += abs(u[e] - u[e - h])
                row.append(total / cnt)
                last_valid = j
            else:
                row.append(row[last_valid] * math.sqrt(h / GRID[last_valid]))
        means.append(row)
    return ew, means


def interpolate(values, horizon):
    """Linear interpolation in h of grid values, for h = 1..horizon; hold the last value beyond the grid."""
    out = []
    j = 0
    last = len(GRID) - 1
    for h in range(1, horizon + 1):
        if h >= GRID[last]:
            out.append(values[last])
            continue
        while GRID[j + 1] < h:
            j += 1
        h0 = GRID[j]
        h1 = GRID[j + 1]
        t = (h - h0) / (h1 - h0)
        out.append(values[j] + t * (values[j + 1] - values[j]))
    return out


def finish_row(values, fallback):
    """Clamp to [0, 1], round to 6 decimals and make the row non-decreasing."""
    row = []
    run = 0.0
    for v in values:
        if not math.isfinite(v):
            v = fallback
        v = round(min(1.0, max(0.0, v)), 6)
        if v < run:
            v = run
        run = v
        row.append(v)
    return row


def fan_rows(x, o75, o95):
    """Persistence median with additive upside and multiplicative downside offsets."""
    rows = []
    xs = max(x, 1e-9)
    for h in range(len(o75)):
        a = o75[h]
        b = o95[h]
        if x <= 0:
            q05 = 0.0
            q25 = 0.0
        else:
            q05 = xs * math.exp(-b / xs)
            q25 = xs * math.exp(-a / xs)
        rows.append(finish_row((q05, q25, x, x + a, x + b), x))
    return rows


def fallback_rows(x, horizon):
    o75 = [Z75 * FALLBACK_SIGMA * math.sqrt(h) for h in range(1, horizon + 1)]
    o95 = [Z95 * FALLBACK_SIGMA * math.sqrt(h) for h in range(1, horizon + 1)]
    return fan_rows(x, o75, o95)


def mature_rows(p, horizon):
    x = p[-1]
    u = usable_history(p)
    if len(u) < MIN_POINTS:
        return fallback_rows(x, horizon)
    ew, means = volatility_features(u)
    g75 = []
    g95 = []
    for j, h in enumerate(GRID):
        rh = math.sqrt(h)
        a75 = 0.0
        a95 = 0.0
        for wi in range(len(WINDOWS)):
            a75 += COEF_75_MEAN[wi] * means[wi][j]
            a95 += COEF_95_MEAN[wi] * means[wi][j]
        for li in range(len(HALF_LIVES)):
            a75 += COEF_75_EWMA[li] * ew[li] * rh
            a95 += COEF_95_EWMA[li] * ew[li] * rh
        g75.append(a75)
        g95.append(max(a95, a75))
    return fan_rows(x, interpolate(g75, horizon), interpolate(g95, horizon))


def newborn_rows(x, horizon):
    rows = []
    xs = max(x, 1e-9)
    for h in range(1, horizon + 1):
        hh = (h / 240.0) ** NEWBORN_BETA
        q05 = xs * math.exp(-NEWBORN_K95D * hh)
        q25 = xs * math.exp(-NEWBORN_K75D * hh)
        q75 = x + NEWBORN_K75U * x * hh
        q95 = x + NEWBORN_K95U * x * hh
        rows.append(finish_row((q05, q25, x, q75, q95), x))
    return rows


def median(values):
    s = sorted(values)
    k = len(s)
    mid = k // 2
    if k % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def young_rows(p, horizon, total_last):
    x = p[-1]
    n = len(p)
    u = usable_history(p)
    if len(u) < MIN_POINTS:
        return newborn_rows(x, horizon)
    group = YOUNG_GROUPS[-1]
    for g in YOUNG_GROUPS:
        if n < g["max_n"]:
            group = g
            break
    ew, means = volatility_features(u)

    dev5 = median(p[-5:]) - x
    dev30 = median(p[-30:]) - x
    dev120 = median(p[-120:]) - x
    sum1 = (1.0 - total_last) * x / total_last if total_last > 0 else 0.0
    gm = group["gamma"]

    g_mid = []
    g_u75 = []
    g_u95 = []
    g_d25 = []
    g_d05 = []
    for j, h in enumerate(GRID):
        rh = math.sqrt(h)
        feats = [means[0][j], means[1][j], means[2][j],
                 ew[0] * rh, ew[1] * rh, ew[2] * rh,
                 0.01 * min(x, 1.0 - x) * rh, 1e-4 * rh]
        u75 = sum(c * f for c, f in zip(group["u75"], feats))
        u95 = sum(c * f for c, f in zip(group["u95"], feats))
        d25 = sum(c * f for c, f in zip(group["d25"], feats))
        d05 = sum(c * f for c, f in zip(group["d05"], feats))
        g_u75.append(u75)
        g_u95.append(max(u95, u75))
        g_d25.append(d25)
        g_d05.append(max(d05, d25))

        s1 = math.sqrt(h / 240.0)
        s2 = h / 240.0
        mg = (gm[0] * x * s1 + gm[1] * x * s2
              + gm[2] * dev5 * s1 + gm[3] * dev30 * s1 + gm[4] * dev120 * s1
              + gm[5] * sum1 * s1 + gm[6] * sum1 * s2
              + gm[7] * dev5 * s2 + gm[8] * dev30 * s2 + gm[9] * dev120 * s2)
        g_mid.append(max(mg, -0.9 * x))

    mid = interpolate(g_mid, horizon)
    u75 = interpolate(g_u75, horizon)
    u95 = interpolate(g_u95, horizon)
    d25 = interpolate(g_d25, horizon)
    d05 = interpolate(g_d05, horizon)
    rows = []
    for h in range(horizon):
        mm = x + mid[h]
        ms = max(mm, 1e-9)
        q05 = ms * math.exp(-d05[h] / ms)
        q25 = ms * math.exp(-d25[h] / ms)
        rows.append(finish_row((q05, q25, mm, mm + u75[h], mm + u95[h]), x))
    return rows


def forecast_market(p, horizon, total_last):
    if not p:
        return [[0.5] * 5 for _ in range(horizon)]
    try:
        if len(p) >= YOUNG_N:
            return mature_rows(p, horizon)
        return young_rows(p, horizon, total_last)
    except Exception:
        return fallback_rows(p[-1], horizon)


def read_horizon(case):
    h = case.get("horizon_minutes") if isinstance(case, dict) else None
    if isinstance(h, int) and not isinstance(h, bool) and 0 < h <= 10000:
        return h
    return DEFAULT_HORIZON


def solve_case(text):
    try:
        case = json.loads(text)
    except Exception:
        return {"forecasts": []}
    horizon = read_horizon(case)
    markets = case.get("markets") if isinstance(case, dict) else None
    if not isinstance(markets, list):
        markets = []
    prices = []
    for market in markets:
        try:
            prices.append(sanitise(market.get("prices") if isinstance(market, dict) else None))
        except Exception:
            prices.append([])
    total_last = sum(p[-1] for p in prices if p)
    forecasts = []
    for market, p in zip(markets, prices):
        mid = market.get("id") if isinstance(market, dict) else None
        try:
            q = forecast_market(p, horizon, total_last)
        except Exception:
            q = [[0.5] * 5 for _ in range(horizon)]
        forecasts.append({"id": mid, "q": q})
    return {"forecasts": forecasts}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    try:
        names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    except Exception:
        names = []
    for name in names:
        case_id = name[:-3]
        try:
            with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
                text = fh.read()
            result = solve_case(text)
        except Exception:
            result = {"forecasts": []}
        try:
            data = json.dumps(result, separators=(",", ":")) + "\n"
        except Exception:
            data = '{"forecasts":[]}\n'
        try:
            with open(os.path.join(out_dir, case_id + ".out"), "w", encoding="utf-8", newline="\n") as fh:
                fh.write(data)
        except Exception as exc:
            print("failed to write %s: %s" % (case_id, exc), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
