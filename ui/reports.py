"""Present the evaluator’s verified benchmark assets and cohort notes."""

from html import escape

from ui.replay import read_json, read_object


METHODS = {
    "random": "Random",
    "classifier": "WHISPERWATCH classifier",
    "native": "Native Whisper confidence",
    "original_hybrid": "50/50 hybrid",
    "control": "Control",
    "audio_aware": "Audio-aware",
    "control_hybrid": "Control hybrid",
    "audio_hybrid": "Audio-aware hybrid",
}
CHARTS = (
    ("baseline_comparison.png", "Review methods at a 20% budget"),
    ("cross_dataset_performance.png", "Performance across evaluation cohorts"),
    ("review_budget_curve.png", "Error recall at different review budgets"),
    ("confusion_matrix.png", "50/50 hybrid confusion matrix on the primary AMI holdout"),
)


def verified_benchmarks(path):
    """Return a summary and table for the measured comparisons."""
    try:
        rows = read_json(path)
        if not isinstance(rows, list) or not rows:
            raise ValueError("Expected benchmark rows.")
        comparisons = []
        table = []
        for row in rows:
            name = escape(row["experiment_name"].partition(" — ")[-1])
            role = escape(row["dataset_role"])
            method = escape(METHODS.get(row["method"], row["method"]))
            total = row["total_clips"]
            reviewed = row["reviewed_clips"]
            caught = row["high_error_caught"]
            errors = row["high_error_total"]
            precision = row["precision_at_budget"]
            recall = row["recall_at_budget"]
            f1 = 2 * caught / (reviewed + errors) if reviewed + errors else 0
            table.append(
                f'<tr><th scope="row">{name}</th><td>{method}</td>'
                f'<td>{reviewed:,} / {total:,}</td><td>{caught:,} / {errors:,}</td>'
                f'<td>{precision:.1%}</td><td>{recall:.1%}</td><td>{f1:.1%}</td></tr>'
            )
            if row["method"] == "random" and caught:
                native = row["native_caught_baseline"]
                comparisons.append(
                    f"<li><strong>{name}</strong> {native} vs {caught} high-error clips caught "
                    f"({native / caught:.2f}× as many). {role}</li>"
                )
        return (
            '<section class="benchmark-summary"><h3>What the Results Show</h3>'
            '<p>Confidence-based review found more high-error clips than random review in '
            'both holdouts with a reported random baseline. The custom classifier performed '
            'comparably to native Whisper confidence. Its additional benefit remains uncertain.</p>'
            f'<ul>{"".join(comparisons)}</ul></section>',
            '<div class="benchmark-table" tabindex="0" role="region" aria-label="Verified benchmarks">'
            '<table><caption>Verified Methods at the Reported Review Budget</caption><thead><tr>'
            '<th scope="col">Cohort</th><th scope="col">Method</th><th scope="col">Reviewed</th>'
            '<th scope="col">High-error clips caught</th><th scope="col">Precision</th>'
            '<th scope="col">Recall</th><th scope="col">F1</th></tr></thead><tbody>'
            f'{"".join(table)}</tbody></table></div>'
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return '<p class="status warning">Verified benchmark results are unavailable.</p>', ""


def dataset_limits(path):
    """Render cohort metadata and safeguards from the evaluator’s notes."""
    try:
        data = read_object(path)
        items = "".join(
            f'<li><strong>{escape(row["name"])}</strong> · {row["clips"]:,} clips, '
            f'{row["high_error_clips"]:,} high-error clips. {escape(row["limitation"])}</li>'
            for row in data["cohorts"]
        )
        safeguards = "".join(f'<li>{escape(item)}</li>' for item in data["methodological_safeguards"])
        return (
            f'<div class="caveat-details"><p>{escape(data["evaluation_purpose"])}</p>'
            f'<ul>{items}</ul><h3>Methodological Safeguards</h3><ul>{safeguards}</ul></div>'
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return '<p class="status warning">The dataset notes are unavailable.</p>'
