import re
from src.config import DATABASE_SYSTEM_MESSAGE

def format_question(question:str, a:str, b:str) -> str:
    return f"{question}\n\nA) {a}\nB) {b}"

def contains_answer(text, answer):
    pattern = rf"(?<!\w){re.escape(str(answer).strip())}(?!\w)"
    return re.search(pattern, text, re.IGNORECASE) is not None

def DPO_format(
    prompt:str,
    chosen:str,
    rejected:str,
    system:str=DATABASE_SYSTEM_MESSAGE
) -> dict:
    formatted = {
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}]
    }
    return formatted