"""Install and verify the classifier used by the live demo."""

import argparse
from pathlib import Path

from .inference import install_production_model, load_production_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_zip", type=Path, help="Team-provided classifier ZIP")
    args = parser.parse_args()
    path = install_production_model(args.model_zip)
    load_production_model()
    print(f"Verified WhisperWatch classifier ready at {path}")


if __name__ == "__main__":
    main()
