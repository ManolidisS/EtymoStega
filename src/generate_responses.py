import re
from src.config import DATABASE_SYSTEM_MESSAGE

def format_question(question:str, a:str, b:str) -> str:
    """
    Given a multiple-choice queston with options A and B, returns a formatted version of the question.
    """
    return f"{question}\n\nA) {a}\nB) {b}"

def contains_answer(text:str, answer:str) -> bool:
    """
    Returns a boolean to show if answer is in text.
    """
    pattern = rf"(?<!\w){re.escape(str(answer).strip())}(?!\w)"
    return re.search(pattern, text, re.IGNORECASE) is not None


def normalize_text(text:str) -> str:
    """
    Removes whitespace.
    """
    return re.sub(r"\s+", " ", text).strip()

def deduplicate_completions(entries:list[dict]) -> list[dict]:
    """
    Removes duplicate entries from the comlpetions list.
    """
    seen = set()
    deduped = []

    for entry in entries:
        key = (
            normalize_text(entry["question"]),
            normalize_text(entry["output"]),
        )

        if key in seen:
            continue

        seen.add(key)
        deduped.append(entry)

    return deduped