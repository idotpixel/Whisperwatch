"""Install and verify the classifier used by the live demo."""

import argparse
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from .inference import PRODUCTION_MODEL_PATH, install_production_model, load_production_model


MODEL_URL = (
    "https://github.com/idotpixel/Whisperwatch/releases/download/"
    "demo-model-v1/whisperwatch-model.zip"
)
MAX_DOWNLOAD_BYTES = 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_zip", type=Path, nargs="?", help="Use a local ZIP instead of downloading the release")
    args = parser.parse_args()
    if args.model_zip is not None:
        install_production_model(args.model_zip)
    elif not PRODUCTION_MODEL_PATH.is_file():
        try:
            with urlopen(MODEL_URL, timeout=30) as response:
                archive = response.read(MAX_DOWNLOAD_BYTES + 1)
        except URLError as error:
            raise RuntimeError("Could not download the classifier. Retry or pass a local model ZIP.") from error
        if len(archive) > MAX_DOWNLOAD_BYTES:
            raise ValueError("The classifier download is larger than expected.")
        install_production_model(BytesIO(archive))
    load_production_model()
    print(f"Verified WhisperWatch classifier ready at {PRODUCTION_MODEL_PATH}")


if __name__ == "__main__":
    main()
