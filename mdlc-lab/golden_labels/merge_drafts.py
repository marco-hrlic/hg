"""Merge two independent draft labellings into draft golden labels plus a human-review sheet.

    python golden_labels/merge_drafts.py <dir with A_chunk*.jsonl and B_chunk*.jsonl>

Writes:
    golden_labels/draft_labels.csv   committed: labels only (item_id, both raters, status), no review text
    data/golden/review_sheet.csv     not committed: the items a person must review, with text

Status per item:
    agreed           both raters chose the same sentiment label with high confidence
    agreed_low_conf  same label, but at least one rater was unsure. Included in the draft golden set,
                     listed for a lighter spot-check. (Requiring high confidence from both would drop
                     nearly every neutral item: the hard cases are exactly the ones raters are unsure of.)
    needs_review     the raters chose different labels. Excluded until a person decides.
    ambiguous        both raters said ambiguous. Excluded unless a reviewer decides otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from mdlc import data, metrics  # noqa: E402


def read_rater(folder: Path, rater: str) -> pd.DataFrame:
    rows = []
    for f in sorted(folder.glob(f"{rater}_chunk*.jsonl")):
        rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows).drop_duplicates("item_id", keep="last")
    return df.rename(columns={"label": f"label_{rater}", "confidence": f"conf_{rater}", "note": f"note_{rater}"})


def main(folder: Path) -> None:
    cands = data.load_golden_candidates()
    a, b = read_rater(folder, "A"), read_rater(folder, "B")
    df = cands[["item_id", "star_label"]].merge(a, on="item_id", how="left").merge(b, on="item_id", how="left")
    missing = df[df.label_A.isna() | df.label_B.isna()]
    if len(missing):
        raise SystemExit(f"{len(missing)} items lack a label from one rater: {missing.item_id.tolist()[:10]}")

    same = df.label_A == df.label_B
    confident = (df.conf_A == "high") & (df.conf_B == "high")
    df["status"] = "needs_review"
    df.loc[same, "status"] = "agreed_low_conf"
    df.loc[same & confident, "status"] = "agreed"
    df.loc[same & (df.label_A == "ambiguous"), "status"] = "ambiguous"
    df["draft_label"] = df.label_A.where(same, "")

    both = df[df.label_A.isin(metrics.LABELS) & df.label_B.isin(metrics.LABELS)]
    print(f"Cohen's kappa A vs B (items both labelled with a sentiment): "
          f"{metrics.cohen_kappa(both.label_A, both.label_B):.3f} on {len(both)} items")
    print(df.status.value_counts().to_string(), "\n")
    agreed = df[df.status.str.startswith("agreed")]
    print(f"agreed labels match star labels on {(agreed.draft_label == agreed.star_label).mean():.0%} of items")

    cols = ["item_id", "label_A", "conf_A", "note_A", "label_B", "conf_B", "note_B", "status", "draft_label"]
    df[cols].to_csv(HERE / "draft_labels.csv", index=False)

    priority = {"needs_review": 0, "ambiguous": 1, "agreed_low_conf": 2}
    review = (df[df.status != "agreed"].merge(cands[["item_id", "text"]], on="item_id")
              .sort_values("status", key=lambda s: s.map(priority)))
    review = review.assign(final_label="", reviewer="", reviewer_note="")
    out = data.DATA_DIR / "golden" / "review_sheet.csv"
    review[["item_id", "text", *cols[1:], "final_label", "reviewer", "reviewer_note"]].to_csv(out, index=False)
    print(f"\nwrote {HERE / 'draft_labels.csv'}\nwrote {out} ({len(review)} items to review)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
