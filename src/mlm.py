from src.config import MLM_NAME, MLM_DOWNLOAD_PATH, TOP_K_MLM
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from src.spacy_code import exact_first_verb
from src.authentication import signin

signin()

tokenizer = AutoTokenizer.from_pretrained(MLM_NAME)
model = AutoModelForMaskedLM.from_pretrained(MLM_NAME)

tokenizer.save_pretrained(MLM_DOWNLOAD_PATH)
model.save_pretrained(MLM_DOWNLOAD_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

mask = tokenizer.mask_token

def fill_mask(text:str, top_k:int=TOP_K_MLM) -> list[dict]:
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits

    mask_positions = (
        inputs["input_ids"][0] == tokenizer.mask_token_id
    ).nonzero(as_tuple=True)[0]

    if len(mask_positions) != 1:
        raise ValueError("Text must contain exactly one mask token.")

    pos = mask_positions[0]
    log_probs = torch.log_softmax(logits[0, pos], dim=-1)
    values, indices = torch.topk(log_probs, top_k)

    results = []
    for log_prob, token_id in zip(values, indices):
        token = tokenizer.decode(
            [token_id.item()],
            clean_up_tokenization_spaces=False
        ).strip()
        filled_text = text.replace(tokenizer.mask_token, token, 1)
        results.append({
            "generation": filled_text,
            "generated_token": token,
            "score": log_prob.item()
        })

    return results

def mask_first_verb(text:str) -> str:
    first_verb = exact_first_verb(text)
    if first_verb:
        return text.replace(first_verb, mask, 1)
    return text