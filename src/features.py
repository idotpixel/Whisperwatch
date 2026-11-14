"""Build classifier inputs from Whisper output and the audio duration."""

from collections.abc import Mapping
from math import isfinite, nan
from statistics import mean


FEATURE_NAMES = (
    "avg_logprob",
    "min_logprob",
    "mean_no_speech_prob",
    "max_no_speech_prob",
    "avg_compression_ratio",
    "num_segments",
    "duration_s",
    "num_words",
    "num_chars",
    "words_per_second",
)


def finite_or_nan(value) -> float:
    """Return a finite number or NaN for the model's missing-value imputer."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return nan
    return number if isfinite(number) else nan


def extract_features(segments, transcript: str, duration_s: float) -> dict:
    """Accept archived segment dictionaries or faster-whisper Segment objects.

    Average each confidence measurement over segments where it is available.
    Count whitespace-delimited words and characters in the stripped transcript.
    Missing measurements and rates with invalid duration are represented as NaN.
    """
    measurements = {name: [] for name in ("avg_logprob", "no_speech_prob", "compression_ratio")}
    num_segments = 0
    for segment in () if segments is None else segments:
        if segment is None:
            continue
        num_segments += 1
        for name, values in measurements.items():
            value = segment.get(name) if isinstance(segment, Mapping) else getattr(segment, name, None)
            number = finite_or_nan(value)
            if isfinite(number):
                values.append(number)

    logprobs = measurements["avg_logprob"]
    no_speech = measurements["no_speech_prob"]
    ratios = measurements["compression_ratio"]
    transcript = transcript.strip() if isinstance(transcript, str) else ""
    duration = finite_or_nan(duration_s)
    num_words = len(transcript.split())
    return {
        "avg_logprob": mean(logprobs) if logprobs else nan,
        "min_logprob": min(logprobs, default=nan),
        "mean_no_speech_prob": mean(no_speech) if no_speech else nan,
        "max_no_speech_prob": max(no_speech, default=nan),
        "avg_compression_ratio": mean(ratios) if ratios else nan,
        "num_segments": num_segments,
        "duration_s": duration if duration >= 0 else nan,
        "num_words": num_words,
        "num_chars": len(transcript),
        "words_per_second": finite_or_nan(num_words / duration) if duration > 0 else nan,
    }
