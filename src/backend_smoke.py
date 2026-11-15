"""Quick backend checks using genuine saved live features and a stub Whisper decoder."""

import json
import math
import os
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from . import asr
from .features import FEATURE_NAMES, extract_features
from .inference import PROJECT_ROOT, load_production_model, predict_from_features, rank_recordings


class BackendSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = PROJECT_ROOT / "notebooks" / "fixtures" / "backend_smoke.json"
        cls.clips = json.loads(fixture.read_text(encoding="utf-8"))["clips"]

    def test_frozen_model_schema_and_wrong_file_guard(self):
        model = load_production_model()
        self.assertEqual(tuple(model["pipeline"].feature_names_in_), FEATURE_NAMES)
        self.assertEqual(model["training_rows"], 1_085_612)
        self.assertEqual(list(model["pipeline"].classes_), [0, 1])
        with TemporaryDirectory() as temp:
            bad = Path(temp) / "candidate.joblib"
            bad.write_bytes(b"other model bytes")
            with self.assertRaisesRegex(ValueError, "original frozen boosted"):
                load_production_model(bad)

    def test_live_objects_archived_records_and_actual_fixture_scores(self):
        for clip in self.clips:
            inputs = (clip["transcript"], clip["duration_s"])
            features = extract_features(clip["segments"], *inputs)
            live = extract_features([SimpleNamespace(**segment) for segment in clip["segments"]], *inputs)
            self.assertEqual(tuple(features), FEATURE_NAMES)
            self.assertEqual(features, live)
            result = predict_from_features(features)
            self.assertAlmostEqual(result["review_score"], clip["expected_score"], places=12)
            self.assertEqual(result["details"]["score_type"], "single_clip_classifier_risk")
            json.dumps(result, allow_nan=False)

    def test_higher_risk_comes_first_and_batch_budget_is_stable(self):
        values = [("e", .1, -.1), ("d", .9, -.1), ("c", .3, -.2), ("b", .4, -.9), ("a", .4, -.9)]
        rows = [{"clip_id": key, "review_score": score, "details": {"avg_logprob": logprob}}
                for key, score, logprob in values]
        native = rank_recordings(rows, method="native_confidence")
        classifier = rank_recordings(rows, method="classifier")
        hybrid = rank_recordings(rows)
        self.assertEqual(native[0]["clip_id"], "a")
        self.assertEqual(classifier[0]["clip_id"], "d")
        self.assertEqual(hybrid[0]["clip_id"], "a")
        self.assertAlmostEqual(hybrid[0]["hybrid_risk"], .8)
        self.assertEqual(sum(row["selected_for_review"] for row in hybrid), 1)
        self.assertEqual(hybrid, rank_recordings(reversed(rows)))
        rows[0]["details"]["avg_logprob"] = None
        self.assertIsNone(rank_recordings(rows, method="native_confidence")[0]["native_confidence_risk"])

    def test_empty_invalid_and_reference_fields(self):
        self.assertEqual(rank_recordings([]), [])
        with self.assertRaises(ValueError):
            rank_recordings([{"clip_id": "bad", "review_score": 4, "details": {"avg_logprob": None}}])
        valid = {"clip_id": "same", "review_score": .5, "details": {"avg_logprob": None}}
        with self.assertRaises(ValueError):
            rank_recordings([valid, valid])
        with self.assertRaises(ValueError):
            rank_recordings([], method="bad")
        empty = extract_features([], "", 0)
        self.assertTrue(math.isnan(empty["words_per_second"]))
        for result in (predict_from_features(empty), predict_from_features({})):
            self.assertTrue(0 <= result["review_score"] <= 1)
            self.assertIsNone(result["details"]["avg_logprob"])
        contaminated = dict(empty, human_text="private reference", wer=1, true_high_error=1, alignment_score=.5)
        result = predict_from_features(contaminated)
        self.assertEqual(result, predict_from_features(empty))
        json.dumps(result, allow_nan=False)

    def test_analyze_contract_with_stub_decoder(self):
        clip = next(clip for clip in self.clips if clip["segments"])
        segments = [SimpleNamespace(**segment) for segment in clip["segments"]]
        with TemporaryDirectory() as temp, patch.dict(os.environ, {"WHISPERWATCH_ASR_DEVICE": "cpu",
                                                                 "WHISPERWATCH_ASR_COMPUTE_TYPE": "float32"}):
            path = Path(temp) / "fixture.wav"
            with self.assertRaises(FileNotFoundError):
                asr.transcribe_audio(str(path))
            path.touch()
            with self.assertRaisesRegex(ValueError, "empty"):
                asr.transcribe_audio(str(path))
            with wave.open(str(path), "wb") as file:
                file.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
                file.writeframes(bytes(32_000))
            with patch.object(asr, "_whisper_model") as model:
                decoder = model.return_value.transcribe
                decoder.return_value = (iter(segments), SimpleNamespace(duration=clip["duration_s"]))
                result = asr.analyze_audio(str(path))
                self.assertEqual(set(result), {"transcript", "review_score", "review_label", "details"})
                self.assertEqual(result["transcript"], clip["transcript"])
                self.assertAlmostEqual(result["review_score"], clip["expected_score"], places=12)
                self.assertEqual(decoder.call_args.kwargs["beam_size"], 5)
                self.assertFalse(decoder.call_args.kwargs["vad_filter"])
                json.dumps(result, allow_nan=False)
                decoder.return_value = (iter(segments), SimpleNamespace(duration=31))
                with self.assertRaisesRegex(ValueError, "up to 30 seconds"):
                    asr.analyze_audio(str(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
