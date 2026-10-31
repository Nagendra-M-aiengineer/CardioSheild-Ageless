from pathlib import Path
from shutil import copy2


PROJECT_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = PROJECT_DIR / "public"


def copy_tree_files(source_dir: Path, destination_dir: Path, pattern: str) -> None:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Required deployment assets not found: {source_dir}")

    for source in source_dir.glob(pattern):
        if source.is_file():
            destination = destination_dir / source.relative_to(source_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy2(source, destination)


def main() -> None:
    PUBLIC_DIR.mkdir(exist_ok=True)

    for pattern in ("*.html", "*.css", "*.js"):
        copy_tree_files(PROJECT_DIR, PUBLIC_DIR, pattern)

    copy_tree_files(PROJECT_DIR / "images", PUBLIC_DIR / "images", "*")
    copy_tree_files(PROJECT_DIR / "model" / "shap", PUBLIC_DIR / "model" / "shap", "*.png")

    print(f"Prepared static deployment assets in {PUBLIC_DIR}")


if __name__ == "__main__":
    main()
