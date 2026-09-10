from src.config import (
    SEED,
    PROJECT_ROOT,
    ETYMOLOGY_DIR,
    DATABASE_DIR,
    create_directories,
    WIKTEXTRACT_URL,
    WIKTEXTRACT_GZ,
    WIKTEXTRACT_JSONL,
    QUESTIONS_TRAINING,
    QUESTIONS_EVALUATION,
    RAW_QUESTIONS
)
import gzip
import shutil
import subprocess
import sys
import random
import json
from pathlib import Path
from urllib.request import urlretrieve

random.seed(SEED)

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

def setup_questions(path_to_questions, path_to_train, path_to_eval):
    with open(path_to_questions, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    random.shuffle(data)
    
    train = data[:1125]
    evalu = data[1125:]

    with open(path_to_train, "w", encoding="utf-8") as f:
        for item in train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(path_to_eval, "w", encoding="utf-8") as f:
        for item in evalu:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def setup():
    print("Creating experiment directories...")
    create_directories()

    setup_questions(
        RAW_QUESTIONS,
        QUESTIONS_TRAINING,
        QUESTIONS_EVALUATION
    )

    print("\nDownloading required files...")

    download(
        WIKTEXTRACT_URL,
        WIKTEXTRACT_GZ,
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
    setup()