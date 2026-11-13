"""Check forecast outputs against the bounty's shape rules.

Usage: python validate.py <input_dir> <output_dir>
Exits non-zero on any violation.
"""

import json
import math
import os
import sys

MAX_BYTES = 1048576
ROWS = 240


def check_case(in_path, out_path):
    errors = []
    with open(in_path, "r", encoding="utf-8") as fh:
        case = json.load(fh)
    ids = [m.get("id") for m in case.get("markets", [])]
    if not os.path.exists(out_path):
        return ["missing output file"]
    size = os.path.getsize(out_path)
    if size > MAX_BYTES:
        errors.append("file is %d bytes, limit %d" % (size, MAX_BYTES))
    with open(out_path, "r", encoding="utf-8") as fh:
        try:
            out = json.load(fh)
        except Exception as exc:
            return errors + ["invalid JSON: %s" % exc]
    if not isinstance(out, dict) or list(out.keys()) != ["forecasts"]:
        return errors + ["top level must be an object with exactly the key 'forecasts'"]
    forecasts = out["forecasts"]
    if not isinstance(forecasts, list):
        return errors + ["'forecasts' must be a list"]
    seen = []
    for entry in forecasts:
        if not isinstance(entry, dict) or set(entry.keys()) != {"id", "q"}:
            errors.append("entry must have exactly the keys 'id' and 'q'")
            continue
        mid = entry["id"]
        seen.append(mid)
        q = entry["q"]
        if not isinstance(q, list) or len(q) != ROWS:
            errors.append("%s: expected %d rows" % (mid, ROWS))
            continue
        for r, row in enumerate(q):
            if not isinstance(row, list) or len(row) != 5:
                errors.append("%s row %d: expected 5 numbers" % (mid, r))
                break
            bad = False
            for v in row:
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or v > 1:
                    errors.append("%s row %d: value %r not a finite number in [0, 1]" % (mid, r, v))
                    bad = True
                    break
            if bad:
                break
            if any(row[i] > row[i + 1] for i in range(4)):
                errors.append("%s row %d: not non-decreasing" % (mid, r))
                break
    if len(seen) != len(set(seen)):
        errors.append("duplicate market ids in output")
    missing = set(ids) - set(seen)
    extra = set(seen) - set(ids)
    if missing:
        errors.append("missing markets: %s" % sorted(missing))
    if extra:
        errors.append("unexpected markets: %s" % sorted(extra, key=str))
    return errors


def main(argv):
    if len(argv) != 3:
        print("usage: python validate.py <input_dir> <output_dir>", file=sys.stderr)
        return 2
    in_dir, out_dir = argv[1], argv[2]
    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    if not names:
        print("no .in files in %s" % in_dir, file=sys.stderr)
        return 1
    failed = 0
    for name in names:
        case_id = name[:-3]
        try:
            errors = check_case(os.path.join(in_dir, name), os.path.join(out_dir, case_id + ".out"))
        except Exception as exc:
            errors = ["check failed: %s" % exc]
        if errors:
            failed += 1
            print("FAIL %s" % case_id)
            for e in errors[:10]:
                print("  " + e)
        else:
            print("ok   %s" % case_id)
    print("%d of %d cases valid" % (len(names) - failed, len(names)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
