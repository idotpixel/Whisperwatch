"""Run the backend in a fresh Colab process with CUDA libraries available at startup."""

import argparse
import json
import os
from importlib.util import find_spec
from pathlib import Path

from .asr import analyze_audio
from .inference import rank_recordings


def gpu_process_env() -> dict[str, str]:
    """Environment for a child process running either the demo or the future UI."""
    libraries = []
    for name in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            spec = find_spec(name)
        except ModuleNotFoundError:
            spec = None
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("Install NVIDIA cuBLAS and cuDNN 9 for CUDA 12 in Colab.")
        libraries.append(str(Path(next(iter(spec.submodule_search_locations))) / "lib"))
    env = dict(os.environ, WHISPERWATCH_ASR_DEVICE="cuda", WHISPERWATCH_ASR_COMPUTE_TYPE="float16")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [*libraries, env.get("LD_LIBRARY_PATH")]))
    return env


def main():
    parser = argparse.ArgumentParser(description="Analyze short English recordings and rank the batch.")
    parser.add_argument("output", type=Path, help="Path for the JSON integration response")
    parser.add_argument("audio", type=Path, nargs="+", help="Paths to uploaded audio files")
    args = parser.parse_args()
    analyzed = [dict(analyze_audio(str(path)), clip_id=f"{number:03d}-{path.name}")
                for number, path in enumerate(args.audio, start=1)]
    result = {"analysis": analyzed, "batch": rank_recordings(analyzed) if len(analyzed) > 1 else []}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
