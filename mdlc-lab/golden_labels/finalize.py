"""Turn draft labels plus a completed review sheet into the final golden labels.

    python golden_labels/finalize.py [path/to/review_sheet.csv]

For every item in the review sheet with a non-empty `final_label`, that label wins.
Items not on the sheet keep their agreed draft label. Items still undecided are left out.
Writes golden_labels/golden_labels.csv (item_id, label, source), which mdlc.data.load_golden
then prefers over the drafts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from mdlc import LABELS, data  # noqa: E402

VALID = set(LABELS) | {"ambiguous"}


def main(sheet_path: Path) -> None:
    drafts = pd.read_csv(HERE / "draft_labels.csv", keep_default_na=False)
    sheet = pd.read_csv(sheet_path, keep_default_na=False)
    decided = sheet[sheet.final_label.str.strip() != ""].copy()
    decided["final_label"] = decided.final_label.str.strip().str.lower()
    bad = decided[~decided.final_label.isin(VALID)]
    if len(bad):
        raise SystemExit(f"invalid final_label values: {bad[['item_id', 'final_label']].to_dict('records')}")

    out = drafts.loc[drafts.status.str.startswith("agreed"), ["item_id", "draft_label"]]
    out = out.rename(columns={"draft_label": "label"}).assign(source="llm_agreed")
    out = out[~out.item_id.isin(decided.item_id)]
    reviewed = decided[["item_id", "final_label"]].rename(columns={"final_label": "label"}).assign(source="human")
    final = pd.concat([out, reviewed]).sort_values("item_id")
    final.to_csv(HERE / "golden_labels.csv", index=False)

    pending = sheet[(sheet.final_label.str.strip() == "") & (sheet.status.isin(["needs_review", "ambiguous"]))]
    print(final.label.value_counts().to_string())
    print(f"\n{len(reviewed)} human decisions applied; {len(pending)} disputed items still undecided (left out).")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else data.DATA_DIR / "golden" / "review_sheet.csv")
