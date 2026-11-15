"""WHISPERWATCH frontend and Colab launch entry point."""

import argparse
import logging
import os
from html import escape
from pathlib import Path

import gradio as gr

from ui.live import OFFLINE, analyze_batch, analyze_upload, backend_present, error_message
from ui.replay import load_replay
from ui.reports import CHARTS, dataset_limits, verified_benchmarks
from ui.views import live_assessment, ranked_batch, replay_assessment, score_card, status_line

ROOT = Path(__file__).resolve().parent
VERIFIED = ROOT / "assets" / "verified"
STYLES = ROOT / "assets" / "whisperwatch.css"
GRADIO_MAJOR = int(gr.__version__.split(".")[0])
PRIVATE_EVENT = {"api_visibility": "private"} if GRADIO_MAJOR >= 6 else {"api_name": False}
LOGGER = logging.getLogger(__name__)
LIVE_INSTRUCTION = "Upload or record a short English clip, then select Transcribe & Estimate Review Risk."


def hero(*, workspace=False, routed=True):
    """Use a static signal motif so opening the app requires no media requests."""
    heights = [8, 14, 25, 17, 36, 63, 37, 23, 48, 77, 96, 58, 31, 55, 82, 64,
               33, 18, 48, 73, 100, 68, 39, 56, 74, 41, 25, 13, 27, 18, 10]
    signal = "".join(f'<i style="--bar:{height}%"></i>' for height in heights)
    mark_tag = "a" if workspace else "div"
    mark_link = ' href="./" aria-label="WHISPERWATCH home"' if workspace else ""
    home_button = """
        <a class="home-button" href="./" aria-label="Home" title="Home">
          <svg viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2"
               stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">
            <path d="m3 11 9-8 9 8M5.5 9.5V21h4.25v-6.5h4.5V21h4.25V9.5"/>
          </svg>
        </a>
    """ if workspace else ""
    target = "./workspace" if routed else "#workspace-tabs"
    action = (
        f'<a class="landing-cta" href="{target}">Open Workspace <span aria-hidden="true">→</span></a>'
        if not workspace else ""
    )
    return f"""
      <header class="masthead">
        <{mark_tag} class="wordmark"{mark_link}>
          <svg class="brand-icon" viewBox="0 0 154 100" aria-hidden="true" focusable="false">
            <defs>
              <linearGradient id="ww-logo-gradient" x2="1" y2=".35">
                <stop stop-color="#3186f4"/>
                <stop offset="1" stop-color="#174287"/>
              </linearGradient>
            </defs>
            <path fill="url(#ww-logo-gradient)" d="M2 47C23 20 51 1 78 1s56 19 74 46q3 4 0 8c-19 27-47 44-75 44S22 82 2 55q-3-4 0-8Z"/>
            <path fill="none" stroke="#fffaf5" stroke-width="8" stroke-linecap="round"
                  d="M37 46v8m14-15v22m14-30v38m14-50v62m14-50v38m14-30v22m14-15v8"/>
          </svg>
          <span>WHISPERWATCH</span>
        </{mark_tag}>
        {home_button}
      </header>
      <section class="hero">
        <div class="hero-copy">
          <h1>Know what needs<br>a second listen.</h1>
          <p>WHISPERWATCH flags AI-generated transcripts that are most likely to need human review.</p>
          {action}
        </div>
        <div class="hero-visual" aria-hidden="true"><div class="signal">{signal}</div></div>
      </section>
    """



LANDING_DETAILS = """
<section class="landing-explainer" aria-labelledby="landing-heading">
  <h2 id="landing-heading">How WHISPERWATCH works</h2>
  <p class="muted">Start with a short English recording. Two models turn it into a transcript and help you decide what to check.</p>
  <div class="landing-grid">
    <article>
      <h3>Transcribe the audio</h3>
      <p>Whisper listens to the recording and generates a transcript you can read alongside the audio.</p>
    </article>
    <article>
      <h3>Estimate review risk</h3>
      <p>WHISPERWATCH uses signals from Whisper, including how confident it was, to estimate which transcripts may contain errors.</p>
    </article>
    <article>
      <h3>Decide what to review</h3>
      <p>Check one clip, or rank a batch to see which clips to listen to first. You make any corrections yourself.</p>
    </article>
  </div>
</section>
"""

FOOTER = (
    '<footer class="app-footer"><span>WHISPERWATCH</span>'
    '<span>Veritas AI International High School Hackathon 2026</span></footer>'
)


def panel_heading(title, description):
    return f'<div class="panel-heading"><h2>{title}</h2><p>{description}</p></div>'


def evaluation_asset(filename):
    """Find an asset in the team handoff, reports checkout, or packaged frontend."""
    for directory in (ROOT / "team_handoff" / "for_member_b_frontend_ui", ROOT / "reports", VERIFIED):
        path = directory / filename
        if path.is_file():
            return path
    return VERIFIED / filename


def live_tab():
    """Build the single-clip workspace and optional batch ranking queue."""
    can_analyze = backend_present()
    can_rank = backend_present(batch=True)

    def status_for(audio_path=None):
        if not can_analyze:
            return status_line(OFFLINE, "warning")
        message = "Recording ready to analyze." if audio_path else LIVE_INSTRUCTION
        return status_line(message)

    def run_live(audio_path, progress=gr.Progress()):
        if not audio_path:
            return "", score_card(), status_line("Record or upload audio before transcribing.", "warning"), None
        if not can_analyze:
            return "", score_card(), status_line(OFFLINE, "warning"), audio_path
        try:
            progress(0.15, desc="Transcribing and estimating review risk")
            result = analyze_upload(audio_path)
            transcript = result["transcript"]
            message = "Analysis complete." if transcript.strip() else "Whisper returned no speech."
            return transcript, live_assessment(result), status_line(message, "success"), audio_path
        except Exception as error:
            LOGGER.exception("Single upload analysis failed")
            return "", score_card(), status_line(error_message(error), "warning"), audio_path

    def reset_live(audio_path, analyzed_path):
        if not audio_path or audio_path == analyzed_path:
            return gr.skip()
        return "", score_card(), status_for(audio_path), None

    def run_batch(audio_paths, progress=gr.Progress()):
        if not audio_paths:
            return "", status_line("Upload at least one recording to rank.", "warning")
        if not can_rank:
            return "", status_line(OFFLINE, "warning")
        try:
            rows, filenames = analyze_batch(audio_paths, progress)
            return ranked_batch(rows, filenames), status_line("Batch ranked.", "success")
        except Exception as error:
            LOGGER.exception("Batch upload analysis failed")
            return "", status_line(error_message(error), "warning")

    with gr.Tab("Live Transcription & Risk Prioritization", id="live"):
        with gr.Row(elem_classes="workspace-grid"):
            with gr.Column(scale=7, min_width=310, elem_classes="card"):
                gr.HTML(panel_heading("Live Transcription", "Upload or record one English clip up to 30 seconds."))
                live_status = gr.HTML(status_for())
                audio = gr.Audio(
                    sources=["upload", "microphone"], type="filepath", label="Audio recording",
                    elem_id="recording",
                )
                gr.HTML('<p class="muted">Stop your microphone recording before starting transcription.</p>')
                analyze = gr.Button(
                    "Transcribe & Estimate Review Risk", variant="primary",
                    elem_classes="analyze-button", interactive=can_analyze,
                )
                live_text = gr.Textbox(
                    label="Generated Transcription", lines=3, max_lines=8,
                    placeholder="The transcript will appear after analysis.",
                    interactive=False, elem_classes="transcript",
                )
            with gr.Column(scale=4, min_width=280, elem_classes="card"):
                live_risk = gr.HTML(score_card())

        analyzed_clip = gr.State(None)
        live_outputs = [live_text, live_risk, live_status, analyzed_clip]
        audio.change(
            reset_live, [audio, analyzed_clip], live_outputs,
            show_progress="hidden", concurrency_id="upload-analysis", **PRIVATE_EVENT,
        )
        audio.clear(
            lambda: ("", score_card(), status_for(), None),
            outputs=live_outputs, show_progress="hidden",
            concurrency_id="upload-analysis", **PRIVATE_EVENT,
        )
        analyze.click(
            run_live, audio, live_outputs, show_progress="full",
            api_name="analyze_audio", concurrency_id="upload-analysis",
        )

        batch_ready = "" if can_rank else status_line(OFFLINE, "warning")
        with gr.Accordion("Rank a Batch of Recordings", open=False, elem_classes="card"):
            gr.HTML('<p class="muted">Compare clips within a batch to decide what to review first. '
                    'The top 20%, rounded up, enter the review queue.</p>')
            batch_files = gr.File(
                label="Audio recordings", type="filepath", file_count="multiple",
                file_types=["audio"], elem_id="batch-recordings",
            )
            rank = gr.Button("Rank Batch", variant="primary", elem_classes=["analyze-button", "rank-button"])
            batch_status = gr.HTML(batch_ready)
            batch_results = gr.HTML()

        batch_files.change(
            lambda: ("", batch_ready), outputs=[batch_results, batch_status],
            show_progress="hidden", concurrency_id="upload-analysis", **PRIVATE_EVENT,
        )
        rank.click(
            run_batch, batch_files, [batch_results, batch_status],
            show_progress="full", api_name="rank_batch", concurrency_id="upload-analysis",
        )


def replay_tab(path):
    """Browse recorded predictions and reference-based outcomes by corpus."""
    replay = load_replay(path)
    with gr.Tab("Dataset Replay & Historical Case Studies", id="replay"):
        if not replay.examples:
            gr.HTML(panel_heading("Dataset Replay", escape(replay.notice)))
            return

        grouped = {}
        for example in replay.examples.values():
            grouped.setdefault(example.dataset, []).append(example.clip_id)
        datasets = sorted(grouped)
        first = replay.examples[grouped[datasets[0]][0]]

        def select_clip(clip_id):
            example = replay.examples.get(clip_id)
            if example is None:
                raise gr.Error("Select a historical clip.")
            return example.whisper_text, example.human_text, replay_assessment(example)

        def select_dataset(dataset):
            choices = grouped[dataset]
            return gr.update(choices=choices, value=choices[0]), *select_clip(choices[0])

        with gr.Row(elem_classes="workspace-grid"):
            with gr.Column(scale=7, min_width=310, elem_classes="card"):
                gr.HTML(panel_heading("Dataset Replay & Historical Case Studies",
                                      escape(replay.source)))
                with gr.Row():
                    dataset = gr.Dropdown(datasets, value=first.dataset, label="Dataset", interactive=True)
                    clip = gr.Dropdown(
                        grouped[first.dataset], value=first.clip_id, label="Historical clip", interactive=True,
                    )
                transcript = gr.Textbox(
                    first.whisper_text, label="Generated Whisper Transcription",
                    lines=3, max_lines=8, interactive=False, elem_classes="transcript",
                )
                reference = gr.Textbox(
                    first.human_text, label="Human Reference Transcript",
                    lines=3, max_lines=8, interactive=False, elem_classes="transcript",
                )
            with gr.Column(scale=4, min_width=280, elem_classes="card"):
                assessment = gr.HTML(replay_assessment(first))

        replay_outputs = [transcript, reference, assessment]
        dataset.change(
            select_dataset, dataset, [clip, *replay_outputs],
            queue=False, show_progress="hidden", **PRIVATE_EVENT,
        )
        clip.change(
            select_clip, clip, replay_outputs,
            queue=False, show_progress="hidden", preprocess=False, api_name="select_clip",
        )


def benchmarks_tab(path):
    """Render published figures, verified counts, and the evaluation limitations."""
    summary, table = verified_benchmarks(path)
    with gr.Tab("Model Evaluation & Benchmarks", id="benchmarks"):
        with gr.Column(elem_classes="card"):
            gr.HTML(panel_heading("Model Evaluation & Benchmarks",
                                  "Verified comparisons at the reported human review budgets."))
            gr.HTML(summary)
            for offset in (0, 2):
                with gr.Row(elem_classes="chart-row"):
                    for filename, title in CHARTS[offset:offset + 2]:
                        chart = path.parent / "figures" / filename
                        if chart.is_file():
                            gr.Image(value=str(chart), label=title, interactive=False)
                        else:
                            gr.HTML(f'<p class="muted">{escape(title)} is unavailable.</p>')
            if table:
                gr.HTML(table)
            with gr.Accordion("Scientific Caveats & Dataset Limitations", open=False):
                gr.HTML(dataset_limits(path.parent / "dataset_limitations.json"))


def workspace_tabs(replay_path, evaluation_path):
    with gr.Tabs(selected="live", elem_id="workspace-tabs"):
        live_tab()
        replay_tab(replay_path)
        benchmarks_tab(evaluation_path)


def theme_settings():
    return {
        "theme": gr.themes.Base(
            primary_hue="blue", neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        ),
        "css": STYLES.read_text(encoding="utf-8"),
    }


def build_app(replay_path=None, evaluation_path=None):
    """Build on Gradio 4, 5, or 6. Older builds use an in-page workspace link."""
    replay_path = Path(replay_path) if replay_path else evaluation_asset("frontend_examples.json")
    evaluation_path = Path(evaluation_path) if evaluation_path else evaluation_asset("verified_experiments.json")
    routed = callable(getattr(gr.Blocks, "route", None))
    settings = theme_settings() if GRADIO_MAJOR < 6 else {}
    with gr.Blocks(title="WHISPERWATCH · Transcription Review", analytics_enabled=False, **settings) as demo:
        with gr.Column(elem_id="whisperwatch"):
            gr.HTML(hero(routed=routed), elem_classes="page-frame")
            gr.HTML(LANDING_DETAILS, elem_classes="page-frame")
            if not routed:
                workspace_tabs(replay_path, evaluation_path)
            gr.HTML(FOOTER, elem_classes="page-frame")

    if routed:
        with demo.route("Workspace", "/workspace"):
            with gr.Column(elem_id="whisperwatch"):
                gr.HTML(hero(workspace=True), elem_classes="page-frame")
                workspace_tabs(replay_path, evaluation_path)
                gr.HTML(FOOTER, elem_classes="page-frame")
    return demo


def launch(*, share=None, replay_path=None, evaluation_path=None, **kwargs):
    """Launch locally, or create a share URL when running in Google Colab."""
    if share is None:
        share = "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ
    settings = theme_settings() if GRADIO_MAJOR >= 6 else {}
    return build_app(replay_path, evaluation_path).queue().launch(
        share=share, **settings, **kwargs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--share", action="store_true", default=None, help="Create a Gradio share URL.")
    parser.add_argument("--port", type=int, help="Local Gradio server port.")
    parser.add_argument("--replay", type=Path, help="Path to a replay JSON export.")
    parser.add_argument("--evaluation", type=Path, help="Path to verified_experiments.json and adjacent figures.")
    args = parser.parse_args()
    launch(share=args.share, replay_path=args.replay, evaluation_path=args.evaluation, server_port=args.port)
