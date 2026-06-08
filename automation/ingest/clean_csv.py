"""Clean a Robinhood activity-report CSV in place.

Real RH exports append non-tabular junk after the transactions — a blank line
plus a free-text tax disclaimer ("The data provided is for informational
purposes only ...") — and occasionally other stray text. Those lines have a
different column count than the header, so the rule is simple and robust:

    keep the header, and keep only rows whose column count matches the header;
    drop everything else (blank lines, footers, stray prose).

The python csv reader correctly treats quoted embedded newlines (RH puts the
CUSIP on a second line inside the Description field) as part of one field, so
those legitimate rows are preserved. Fields are re-written QUOTE_ALL to match
Robinhood's original all-quoted style.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def clean_activity_csv(path: str | Path) -> dict[str, Any]:
    """Remove non-tabular lines from an RH activity CSV, rewriting it in place.

    Returns a summary: {"kept": int, "removed": int, "removed_samples": list}.
    Safe to call on an already-clean file (it just reports removed=0).
    """
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return {"kept": 0, "removed": 0, "removed_samples": []}

    header = rows[0]
    n_cols = len(header)
    cleaned: list[list[str]] = [header]
    removed: list[list[str]] = []

    for row in rows[1:]:
        if not row or all((cell or "").strip() == "" for cell in row):
            removed.append(row)  # blank line
            continue
        if len(row) != n_cols:
            removed.append(row)  # footer / stray prose with wrong field count
            continue
        cleaned.append(row)

    if removed:
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
            writer.writerows(cleaned)

    return {
        "kept": len(cleaned) - 1,
        "removed": len(removed),
        # First cell of each dropped line, truncated — enough to see what went.
        "removed_samples": [
            (" ".join(c for c in r if c).strip()[:80] or "(blank line)") for r in removed[:5]
        ],
    }


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:]:
        summary = clean_activity_csv(arg)
        print(f"{arg}: kept {summary['kept']}, removed {summary['removed']}")
        for sample in summary["removed_samples"]:
            print(f"  - dropped: {sample}")
