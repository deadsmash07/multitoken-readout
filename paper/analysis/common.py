"""Shared helpers for the paper's offline analyses (CPU only, no model). Every function reads the stored
runs/<tag>/results.json or readouts files and re-derives counts from item rows, so the paper's numbers can be
traced to the files rather than to earlier summaries. Scoring rules are copied from scripts/x3_patchscope.py
(cut / norm) and sjlens/wsbench_regex.py (the vendored WorkspaceBench matcher)."""
import csv, json, math, random, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # sjlens/
OUT = Path(__file__).resolve().parent  # paper/analysis/
sys.path.insert(0, str(ROOT))
from sjlens import wsbench_regex as R  # noqa: E402

STOPS = (";", "->", "\n")  # identity3 stop markers (scripts/x3_patchscope.py STOPS)


def load(tag):
    return json.load(open(ROOT / "runs" / tag / "results.json", encoding="utf-8"))


def block(tag, layer=None):
    r = load(tag)
    L = r["layers"]
    key = str(layer) if layer is not None else next(iter(L))
    return L[key]


def rows(tag, arm, control="none", layer=None, kind=None, position_rule=None):
    out = []
    for row in block(tag, layer)["rows"]:
        if row["arm"] != arm or row["control"] != control:
            continue
        if kind is not None and row.get("kind") != kind:
            continue
        if position_rule is not None and row.get("position_rule") != position_rule:
            continue
        out.append(row)
    return out


def cut(text):
    ends = [text.find(m) for m in STOPS if text.find(m) >= 0]
    return text[: min(ends)] if ends else text


def norm(s):
    return re.sub(r"\s+", " ", s).strip().casefold()


def exact_hits(row):
    """registered rule: cut at the identity stops, normalise, equal the item string (per carrier)."""
    t = norm(row["string"])
    return [norm(cut(g)) == t for g in row["gens"]]


def prefix_hits(row):
    t = norm(row["string"])
    return [norm(g).startswith(t) for g in row["gens"]]


def bench_hits(row):
    """the benchmark's word-boundary matcher for the item string anywhere in the uncut generation (per carrier)."""
    m = R.unicode_word_matcher(str(row["string"]).strip())
    return [bool(m(g)) for g in row["gens"]]


def majority(hits):
    return int(sum(hits) > len(hits) / 2)


def cluster_boot(clusters, n=10000, seed=0):
    """clusters: list of lists of per-row values. Resample clusters with replacement, keep every row of a sampled
    cluster, recompute the row-weighted mean; percentile 95% interval. Returns (mean, lo, hi)."""
    rng = random.Random(seed)
    m = len(clusters)
    sums = [sum(c) for c in clusters]
    lens = [len(c) for c in clusters]
    means = []
    for _ in range(n):
        s = 0.0
        k = 0
        for _ in range(m):
            j = rng.randrange(m)
            s += sums[j]
            k += lens[j]
        means.append(s / k)
    means.sort()
    total = sum(sums) / sum(lens)
    return total, means[int(0.025 * n)], means[int(0.975 * n) - 1]


def item_boot(values, n=10000, seed=0):
    return cluster_boot([[v] for v in values], n=n, seed=seed)


def wilson(k, n, z=1.959964):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def binom_p_ge(k, n, p0):
    """one-sided exact binomial P(X >= k | n, p0)."""
    return sum(math.comb(n, i) * p0**i * (1 - p0) ** (n - i) for i in range(k, n + 1))


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        v = min(1.0, (m - rank) * pvals[i])
        running = max(running, v)
        adj[i] = running
    return adj


def write_json(name, obj):
    with open(OUT / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def write_csv(name, rows_, fields):
    with open(OUT / name, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows_:
            w.writerow(r)


def f3(v):
    return "-" if v is None else f"{v:.3f}"
