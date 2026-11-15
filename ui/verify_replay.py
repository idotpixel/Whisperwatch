"""Check the showcase against Member C's immutable, full-cohort predictions."""

import csv
import subprocess
from itertools import groupby
from math import inf
from pathlib import Path

from ui.replay import parse_replay, read_object
from ui.views import replay_assessment


ROOT = Path(__file__).resolve().parents[1]


def archived_rows(commit, path):
    """Read a CSV from the pinned evaluation commit without changing branches."""
    contents = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
    return list(csv.DictReader(contents.decode("utf-8-sig").splitlines(keepends=True)))


def native_risk(row):
    return -float(row["avg_logprob"]) if row["avg_logprob"] else inf


def percentiles(rows, score):
    """Mirror the historical average-tie percentile ranks for verification only."""
    ranks = {}
    start = 0
    for _, group in groupby(sorted(rows, key=score), key=score):
        tied = list(group)
        stop = start + len(tied)
        rank = (start + 1 + stop) / (2 * len(rows))
        for row in tied:
            ranks[row["clip_id"]] = rank
        start = stop
    return ranks


def verify_cohort(rows, counts):
    by_id = {row["clip_id"]: row for row in rows}
    assert len(by_id) == len(rows), "Duplicate source clip IDs"
    truth = {row["clip_id"] for row in rows if float(row["wer"]) > 0.20}
    assert truth == {row["clip_id"] for row in rows if row["true_high_error"] == "1"}
    native = percentiles(rows, native_risk)
    classifier = percentiles(rows, lambda row: float(row["classifier_risk"]))
    hybrid = {clip_id: (native[clip_id] + classifier[clip_id]) / 2 for clip_id in by_id}
    assert all(hybrid[key] == float(row["original_hybrid_risk"]) for key, row in by_id.items())
    slots = (len(rows) + 4) // 5  # 20% rounded up in these three cohorts.
    for method, score in [("original_hybrid", hybrid.__getitem__),
                          ("native", lambda key: native_risk(by_id[key]))]:
        selected = set(sorted(by_id, key=lambda key: (-score(key), key))[:slots])
        saved = {key for key, row in by_id.items() if row[f"{method}_review"] == "True"}
        assert selected == saved, f"Historical {method} review flags differ from full-cohort ranking"
        reported = counts[method]
        assert reported["budget_rounding"] == "ceil"
        assert (len(rows), len(truth), slots, len(selected & truth)) == tuple(
            int(reported[field]) for field in ("clips", "high_error", "reviewed", "caught")
        )
    return by_id


def verify_case(case, original, rendered):
    archived_fields = {
        "whisper_text": original["transcript"],
        "human_text": original["reference"],
        "wer": float(original["wer"]),
        "true_high_error": int(original["true_high_error"]),
        "risk_score": float(original["original_hybrid_risk"]),
        "classifier_risk": float(original["classifier_risk"]),
        "native_risk": float(original["native_risk"]),
        "avg_logprob": float(original["avg_logprob"]),
        "recommend_review": original["original_hybrid_review"] == "True",
        "native_review": original["native_review"] == "True",
    }
    for field, value in archived_fields.items():
        assert case[field] == value, f"{case['clip_id']}: {field} changed"
    high_error = bool(archived_fields["true_high_error"])
    hybrid_selected = archived_fields["recommend_review"]
    native_selected = archived_fields["native_review"]
    outcome = (
        ("True positive" if hybrid_selected else "False negative") if high_error
        else ("False positive" if hybrid_selected else "True negative")
    )
    assert all(fragment in rendered for fragment in (
        outcome, f"{case['wer']:.1%}", f"{case['risk_score']:.3f}",
        "Selected for review" if hybrid_selected else "Not selected for review",
        "<dd>Selected</dd>" if native_selected else "<dd>Not selected</dd>",
    )), f"{case['clip_id']}: displayed results differ from the archive"
    if hybrid_selected != native_selected:
        assert ("Hybrid only" if hybrid_selected else "Native confidence only") in rendered
    return outcome


def main():
    document = read_object(ROOT / "assets" / "replay_cases.json")
    examples = parse_replay(document).examples
    commit = document["source_commit"]
    verified = archived_rows(commit, "resources/verified_counts.csv")
    cohorts = {}
    for dataset, path in document["prediction_files"].items():
        counts = {row["method"]: row for row in verified if row["file"] == Path(path).stem}
        cohorts[dataset] = verify_cohort(archived_rows(commit, path), counts)
        print(f"{dataset}: both rankings verified across {len(cohorts[dataset]):,} clips")
    for case in document["examples"]:
        clip_id = case["clip_id"]
        original = cohorts[case["dataset"]][clip_id]
        outcome = verify_case(case, original, replay_assessment(examples[clip_id]))
        print(f"{clip_id}: {outcome} | WER {case['wer']:.1%} | hybrid {case['risk_score']:.3f}")
    print(f"Verified all {len(examples)} curated cases.")


if __name__ == "__main__":
    main()
