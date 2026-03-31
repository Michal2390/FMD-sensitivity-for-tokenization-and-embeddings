"""Generate publication plots from paper benchmark artifacts."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.publication_plots import generate_publication_plots
from utils.config import load_config


def main() -> None:
    config = load_config("configs/config.yaml")
    outputs = generate_publication_plots(config)
    if not outputs:
        print("No plots generated. Run `python main.py --mode paper-full` first.")
        return

    print("Generated publication plots:")
    for label, path in outputs.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()

