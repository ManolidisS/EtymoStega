from src.config import (
    PROJECT_ROOT,
    ETYMOLOGY_DIR,
    DATABASE_DIR,
    create_directories,
    WIKTEXTRACT_URL,
    QUESTIONS_TRAINING_URL,
    QUESTIONS_VALIDATION_URL,
    WIKTEXTRACT_GZ,
    WIKTEXTRACT_JSONL,
    QUESTIONS_TRAINING,
    QUESTIONS_VALIDATION
)
import gzip
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve

def download(url: str, destination: Path):
    """Download a file if it does not already exist."""

    if destination.exists():
        print(f"Already exists: {destination}")
        return

    print(f"Downloading {destination.name}...")
    urlretrieve(url, destination)


def extract_gzip(source: Path, destination: Path):
    """Extract a .gz file if the extracted file does not exist."""

    if destination.exists():
        print(f"Already extracted: {destination}")
        return

    print(f"Extracting {source.name}...")

    with gzip.open(source, "rb") as compressed:
        with destination.open("wb") as extracted:
            shutil.copyfileobj(compressed, extracted)


def run_module(module: str, *args: str):
    """Run one of the project's Python modules."""

    print(f"Running {module}...")

    subprocess.run(
        [
            sys.executable,
            "-m",
            module,
            *args,
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

def setup():
    print("Creating experiment directories...")
    create_directories()

    print("\nDownloading required files...")

    download(
        WIKTEXTRACT_URL,
        WIKTEXTRACT_GZ,
    )

    download(
        QUESTIONS_TRAINING_URL,
        QUESTIONS_TRAINING,
    )

    download(
        QUESTIONS_VALIDATION_URL,
        QUESTIONS_VALIDATION,
    )

    print("\nExtracting Wiktextract data...")

    extract_gzip(
        WIKTEXTRACT_GZ,
        WIKTEXTRACT_JSONL,
    )

    print("\nBuilding etymology databases...")

    run_module("src.gen_sql")
    run_module("src.verb_etymology", "--build")

    print("\nSetup complete.")

if __name__ == "__main__":
    create_directories()