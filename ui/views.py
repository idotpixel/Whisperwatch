"""Render the shared review card and evaluation visuals."""

from html import escape
from math import isfinite

from ui.replay import number_field, read_object, text_field


def status_line(message, tone="neutral"):
    return f'<p class="status {tone}" role="status"><i></i>{escape(message)}</p>'


def score_card(
    score=None, review=False, *, title="Estimated Review Risk", recommendation=None, description=None,
):
    """Show the model score as a position on its scale."""
    if score is None:
        return """<div class="risk-empty">
          <span class="panel-label">Review Assessment</span>
          <p class="risk-value">—</p>
          <p class="muted">Analyze an audio recording to see its review score.</p>
        </div>"""
    tone = "warning" if review else "low"
    icon = "!" if review else "✓"
    label = recommendation or ("Review Recommended" if review else "Lower Review Priority")
    explanation = description or (
        "Prioritize this transcript for a human check."
        if review else "Lower priority for review. Errors may still be present."
    )
    return f"""<div class="risk-card {tone}" aria-live="polite">
      <div class="risk-heading"><span class="panel-label">{escape(title)}</span></div>
      <div class="risk-value">{score:.3f}<small> / 1.00</small></div>
      <div class="risk-track" role="meter" aria-label="{escape(title)}"
           aria-valuemin="0" aria-valuemax="1" aria-valuenow="{score}">
        <div class="risk-fill" style="width:{score * 100:.4f}%"></div>
      </div>
      <div class="scale-labels"><span>0 · Lower</span><span>1 · Higher</span></div>
      <div class="recommendation"><span aria-hidden="true">{icon}</span>{escape(label)}</div>
      <p class="muted">{escape(explanation)}</p>
    </div>"""


LIVE_REVIEW_CUTOFF = 0.70


def confidence_text(value):
    """Format the raw Whisper log-probability without rescaling it."""
    return f"{value:.3f}" if type(value) in (float, int) and isfinite(value) else "Unavailable"


def live_assessment(result):
    """Show the single-clip classifier and the native Whisper signal side by side."""
    score = result["review_score"]
    review = score >= LIVE_REVIEW_CUTOFF
    recommendation = "Human Review Recommended" if review else "Low Estimated Risk · Standard Review"
    card = score_card(score, review, recommendation=recommendation)
    logprob = confidence_text(result["details"].get("avg_logprob"))
    return f"""{card}
      <p class="muted cutoff-note">Single-clip review cutoff {LIVE_REVIEW_CUTOFF:.2f}.
        The top-20% review budget applies when ranking a batch.</p>
      <section class="confidence-breakdown" aria-label="Confidence breakdown">
        <h3>Confidence Breakdown</h3>
        <dl>
          <div><dt>Whisper native confidence</dt><dd>{logprob}</dd>
            <small>Mean segment log probability. Values closer to zero indicate higher confidence.</small></div>
          <div><dt>WHISPERWATCH classifier risk</dt><dd>{score:.3f} / 1.00</dd>
            <small>Estimate based on Whisper’s decoding signals and recording duration.</small></div>
        </dl>
      </section>"""


def replay_assessment(example):
    """Present the saved decision and reference-based outcomes for one historical case."""
    has_hybrid = example.classifier_risk is not None
    title = "Historical Hybrid Review Priority" if has_hybrid else "Archived Review Score"
    decision = "Selected for review" if example.recommend_review else "Not selected for review"
    description = (
        "The 0–1 score combines percentile ranks within the original cohort. "
        "Probability calibration is unverified." if has_hybrid else "Recorded decision from the evaluation export."
    )
    card = score_card(example.risk_score, example.recommend_review, title=title,
                      recommendation=decision, description=description)
    truth = "High WER" if example.true_high_error else "Within WER cutoff"
    outcome = {
        (True, True): "True positive",
        (True, False): "False negative",
        (False, True): "False positive",
        (False, False): "True negative",
    }[example.true_high_error, example.recommend_review]
    native_decision = ""
    budget = ""
    disagreement = ""
    if example.native_review is not None:
        native_decision = f"""<div><dt>Native-confidence review</dt>
            <dd>{'Selected' if example.native_review else 'Not selected'}</dd></div>"""
        if example.native_review != example.recommend_review:
            disagreement = " · Hybrid only" if example.recommend_review else " · Native confidence only"
        budget = """<p class="muted cutoff-note">Each method used a 20% review budget in the original cohort.
            The classifier predicts WER-defined error risk. The experiment defines high error as WER above 20%.</p>"""
    classifier = (
        f"<div><dt>WHISPERWATCH classifier risk</dt><dd>{example.classifier_risk:.3f}</dd></div>"
        if example.classifier_risk is not None else ""
    )
    notes = f"""<div class="case-note"><h3>{escape(example.case_title or 'Case Note')}</h3>
        <p>{escape(example.notes)}</p>
        <p style="overflow-wrap:anywhere">Clip ID: {escape(example.clip_id)}</p></div>"""
    return f"""{card}{budget}<dl class="case-outcomes">
      <div><dt>Ground-truth WER</dt><dd>{example.wer:.1%}</dd></div>
      <div><dt>True Error Status</dt><dd>{truth}</dd></div>
      <div><dt>{'Hybrid outcome' if has_hybrid else 'Recorded outcome'}</dt><dd>{outcome}{disagreement}</dd></div>
      {native_decision}
      {classifier}
      <div><dt>Whisper mean log probability</dt><dd>{confidence_text(example.avg_logprob)}</dd></div>
    </dl>{notes}"""


def ranked_batch(rows, filenames):
    """Display the backend's rank and review selection for every uploaded clip."""
    selected_count = sum(row["selected_for_review"] for row in rows)
    parts = []
    for row in rows:
        clip_id = row["clip_id"]
        filename = escape(filenames[clip_id])
        selected = row["selected_for_review"]
        tone = " selected" if selected else ""
        badge = '<span class="pill review">Selected for review</span>' if selected else ""
        transcript = escape(row["transcript"] or "Whisper returned an empty transcript.")
        confidence = row["native_confidence_risk"]
        native_risk = "Unavailable · highest native priority" if confidence is None else f"{confidence:.3f}"
        parts.append(f"""<li class="batch-clip{tone}">
          <div class="batch-title"><span class="batch-rank" aria-label="Rank {row['rank']}">#{row['rank']}</span>
            <div><h3>{filename}</h3><span class="muted">{escape(clip_id)}</span></div>{badge}</div>
          <dl class="batch-scores">
            <div><dt>Hybrid priority</dt><dd>{row['hybrid_risk']:.3f}</dd></div>
            <div><dt>Estimated classifier risk</dt><dd>{row['classifier_risk']:.3f}</dd></div>
            <div><dt>Whisper confidence risk</dt><dd>{native_risk}</dd></div>
          </dl>
          <details><summary>Whisper transcript</summary><p dir="auto">{transcript}</p></details>
        </li>""")
    return f"""<section class="batch-results">
      <h2>Review queue</h2>
      <p class="muted">{selected_count} of {len(rows)} clips selected for human review.</p>
      <ol class="batch-list">{''.join(parts)}</ol>
    </section>"""
