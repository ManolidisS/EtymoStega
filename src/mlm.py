from src.config import MLM_NAME, MLM_DOWNLOAD_PATH, TOP_K_MLM
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from src.spacy_code import exact_first_verb, first_word
from src.authentication import signin

signin()

tokenizer = AutoTokenizer.from_pretrained(MLM_NAME)
model = AutoModelForMaskedLM.from_pretrained(MLM_NAME)

tokenizer.save_pretrained(MLM_DOWNLOAD_PATH)
model.save_pretrained(MLM_DOWNLOAD_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
# model = torch.compile(model) # Omitted because we don't have a C compiler. Uncomment if you do.

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

def fill_mask_batches(
    texts: list[str],
    batch_size: int = 128,
    top_k: int = TOP_K_MLM
) -> list[list[dict]]:
    """
    Processes a list of texts in batches.
    Each text must contain exactly one mask token.
    """
    mask_token_id = tokenizer.mask_token_id
    mask_token_str = tokenizer.mask_token
    all_results = []

    with torch.inference_mode():  # Faster than torch.no_grad()
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Dynamic padding to the longest sequence in this batch
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            input_ids = inputs["input_ids"]
            logits = model(**inputs).logits

            # Vectorized search for the mask token index in each row
            mask_indices = (input_ids == mask_token_id).nonzero(as_tuple=True)
            batch_row_indices = mask_indices[0]
            mask_pos_indices = mask_indices[1]

            # Verify each row has exactly one mask
            if len(batch_row_indices) != len(batch_texts):
                raise ValueError(
                    f"Each text must contain exactly one '{mask_token_str}'."
                )

            # Gather logits at the mask positions: shape (batch_size, vocab_size)
            mask_logits = logits[batch_row_indices, mask_pos_indices]

            # Compute log probabilities and top_k
            log_probs = torch.log_softmax(mask_logits, dim=-1)
            topk_values, topk_indices = torch.topk(log_probs, top_k, dim=-1)

            # Move results to CPU at once
            topk_values = topk_values.cpu().tolist()
            topk_indices = topk_indices.cpu().tolist()

            # Decode results
            for row_idx, text in enumerate(batch_texts):
                row_results = []
                for score, token_id in zip(topk_values[row_idx], topk_indices[row_idx]):
                    token = tokenizer.decode(
                        [token_id],
                        clean_up_tokenization_spaces=False
                    ).strip()
                    row_results.append({
                        "generation": text.replace(mask_token_str, token, 1),
                        "generated_token": token,
                        "score": score
                    })
                all_results.append(row_results)

    return all_results

def mask_first_verb(text:str) -> str:
    first_verb = exact_first_verb(text)
    if first_verb:
        return text.replace(first_verb, mask, 1)
    return text

def mask_first_word(text:str) -> str:
    first_word_ = first_word(text)
    if first_word_:
        return text.replace(first_word_, mask, 1)
    return text