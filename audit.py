"""Recompute the published R2 router on matched 4B and 35B runs. Python 3.10+."""

import argparse
import csv
import hashlib
import json
import random
import ssl
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

RECORD = "6b7caca9bd932d69859d0cc10f97fd9aecff4ce9"
TRACES = "f96e277cf8584b38c35c01106c1fe77457dfaa81"
MANIFEST = "4b5d3b7142bf65f919d145b89ef3ca1669f7773114647e415c4be87864be6791"
DIMS = ("correctness", "completeness", "usefulness")
NO_SOURCE = {"grounding:no-good-source", "grounding:no-source-memory"}


def download(url, target, expected):
    if not target.exists():
        request = Request(url, headers={"User-Agent": "BOAR-routing-reproduction/1.0"})
        bundle = Path("/etc/ssl/cert.pem")
        context = ssl.create_default_context(
            cafile=str(bundle) if bundle.exists() else None
        )
        with urlopen(request, timeout=40, context=context) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Downloaded checksum mismatch for {target.name}")
        target.write_bytes(data)
    data = target.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Cache checksum mismatch for {target.name}")
    return data


def source_reader(cache):
    url = (
        f"https://raw.githubusercontent.com/r4topunk/boar-research/{RECORD}/TRACES.tsv"
    )
    manifest = download(url, cache / "TRACES.tsv", MANIFEST).decode().splitlines()
    hashes = {
        row["path"]: row["sha256"] for row in csv.DictReader(manifest, delimiter="\t")
    }

    def read(path):
        url = f"https://huggingface.co/datasets/r4topunk/boar-traces/resolve/{TRACES}/{path}"
        data = download(url, cache / Path(path).name, hashes[path])
        return [json.loads(line) for line in data.splitlines() if line]

    return read


def arm(read, root, stem, seeds, timings=True):
    scores, latency, categories, reference_hashes = (
        defaultdict(list),
        defaultdict(list),
        {},
        set(),
    )
    signals = defaultdict(set)
    for seed in seeds:
        pair = defaultdict(dict)
        for r in read(f"{root}/{stem}-s{seed}.judgments.jsonl"):
            assert (
                r["ok"]
                and r["judgeVersion"] == "judge.v1"
                and r["judgeModel"] == "claude-opus-5-5"
            )
            key, order = r["queryId"], r["order"]
            assert order not in pair[key]
            pair[key][order] = [
                sum(r["mapped"][party][dim] for dim in DIMS)
                for party in ("boar", "ref")
            ]
            categories[key] = r["category"]
            reference_hashes.add(r["refsSha256"])
        for key, orders in pair.items():
            assert set(orders) == {"boarA", "boarB"}
            scores[key].append(tuple(map(sum, zip(*orders.values()))))
        if timings:
            seen = set()
            for row in read(f"{root}/{stem}-s{seed}.traces.jsonl"):
                key = row["itemId"]
                if key not in pair:
                    continue
                assert not row["error"] and key not in seen
                seen.add(key)
                signals[key].add(
                    bool(NO_SOURCE.intersection(row["route"]["reasonCodes"]))
                )
                latency[key].append(row["timing"]["totalMs"] / 1000)
            assert seen == pair.keys()
    assert all(len(v) == len(seeds) for v in scores.values())
    # Each item has equal weight: two answer orders and three dimensions per seed.
    return (
        {key: tuple(map(sum, zip(*values))) for key, values in scores.items()},
        signals,
        {key: statistics.mean(v) for key, v in latency.items()},
        categories,
        reference_hashes,
    )


def permutation(base, changes, routed, groups, observed, draws):
    rng = random.Random(1520)
    pools = [(ids, len(routed.intersection(ids))) for ids in groups]
    samples = []
    for _ in range(draws):
        chosen = [i for ids, k in pools for i in rng.sample(ids, k)]
        app = base[0] + sum(changes[i][0] for i in chosen)
        ref = base[1] + sum(changes[i][1] for i in chosen)
        samples.append(app / ref)
    exceedances = sum(s >= observed for s in samples)
    samples.sort()
    return {
        "mean": statistics.mean(samples),
        "p_one_sided": (exceedances + 1) / (draws + 1),
        "q025": samples[int(0.025 * draws)],
        "q975": samples[int(0.975 * draws)],
    }


def analyze(read, dataset, treatment, deep_stem, seeds, draws):
    folder = "bench-v02" if dataset == "v02" else "vitalik-para1"
    root = f"rnd/heldout/{folder}"
    cheap, signals, cheap_wall, categories, refs = arm(
        read, root + "/runs-cut6", "cut6-" + dataset, seeds
    )
    timing = treatment == "deepvpara"
    deep, _, deep_wall, deep_cats, deep_refs = arm(
        read, root + "/runs-" + treatment, deep_stem, seeds, timing
    )
    assert (
        cheap.keys() == deep.keys()
        and categories == deep_cats
        and len(refs) == 1
        and refs == deep_refs
    )
    ids = sorted(cheap)
    assert len(ids) == (110 if dataset == "v02" else 51)
    assert all(len(signals[i]) == 1 for i in ids)  # The two-seed signal never flipped.
    routed = {i for i in ids if signals[i] == {True}}
    base = tuple(map(sum, zip(*cheap.values())))
    full = tuple(map(sum, zip(*deep.values())))
    changes = {i: tuple(d - b for d, b in zip(deep[i], cheap[i])) for i in ids}
    hybrid = tuple(base[j] + sum(changes[i][j] for i in routed) for j in (0, 1))
    b, d, h = [app / ref for app, ref in (base, full, hybrid)]
    groups = defaultdict(list)
    for i in ids:
        groups[categories[i]].append(i)
    row = {
        "dataset": dataset,
        "deep_run": treatment,
        "seeds": seeds,
        "n": len(ids),
        "routed": len(routed),
        "cheap": b,
        "deep": d,
        "hybrid": h,
        "capture": (h - b) / (d - b),
        "uniform": permutation(base, changes, routed, [ids], h, draws),
        "category_matched": permutation(
            base, changes, routed, groups.values(), h, draws
        ),
    }
    if timing:
        t0 = statistics.mean(cheap_wall.values())
        t1 = statistics.mean(deep_wall.values())
        th = statistics.mean(
            deep_wall[i] if i in routed else cheap_wall[i] for i in ids
        )
        f = (th - t0) / (t1 - t0)
        app, ref = [(1 - f) * base[j] + f * full[j] for j in (0, 1)]
        row["desktop_seconds"] = {
            "cheap": t0,
            "deep": t1,
            "hybrid": th,
            "random_at_equal_mean_time": app / ref,
        }
    return row


def main(cache, draws):
    cache.mkdir(parents=True, exist_ok=True)
    read = source_reader(cache)
    arms = [
        ("v02", "deepvpara", "dv-D-v02", [42, 43]),
        ("vpara", "deepvpara", "dv-D-vpara", [42, 43]),
        ("v02", "dt512v02", "dhv-H-v02", [42]),
        ("v02", "deepthinkv02", "dtv-TV-v02", [42, 43]),
    ]
    results = [analyze(read, *case, draws) for case in arms]
    print(
        json.dumps(
            {"permutation_draws": draws, "permutation_seed": 1520, "results": results},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        help="Optional directory to reuse checksum-verified downloads",
    )
    parser.add_argument("--draws", type=int, default=50000)
    args = parser.parse_args()
    if args.draws < 100:
        parser.error("Use at least 100 permutations")
    with tempfile.TemporaryDirectory() as scratch:
        main(args.cache or Path(scratch), args.draws)
