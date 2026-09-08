from src.config import VERB_POS
import spacy
spacy.require_cpu()
nlp = spacy.load("en_core_web_lg")

def exact_first_verb(text:str, doc=None) -> str | None:
    if not doc:
        doc = nlp(text)
    for token in doc:
        if token.pos_ in VERB_POS:
            return token.text
    return None

def exact_verbs(text:str, doc=None) -> list:
    if not doc:
        doc = nlp(text)
    return [token.text.lower() for token in doc if token.pos_ in VERB_POS]

def lemmatize(word:str) -> str:
    return nlp(word)[0].lemma_

def lemma_first_verb(text:str, doc=None) -> str | None:
    first_verb = exact_first_verb(text, doc=doc)
    if first_verb:
        return lemmatize(first_verb)
    return None

def lemma_verbs(text:str, doc=None) -> list:
    verbs = exact_verbs(text, doc=doc)
    verbs = [lemmatize(verb) for verb in verbs]
    return verbs

def first_word(text: str, doc=None) -> str:
    if not doc:
        doc = nlp(text)
    for token in doc:
        if token.is_alpha:
            return token.text
    return ""

def is_grammatically_compatible(original_sentence: str, candidate_verb: str, orig_doc=None) -> bool:
    if not orig_doc:
        orig_doc = nlp(original_sentence.strip())
    orig_first_token = orig_doc[0]
    target_tag = orig_first_token.tag_  # e.g., 'VBG' or 'VB'

    tokens = [t.text for t in orig_doc]
    new_opening = f"{candidate_verb.capitalize()} " + " ".join(tokens)
    cand_doc = nlp(new_opening)
    cand_first_token = cand_doc[0]
    
    # Heuristic
    if target_tag == "VBG" and not candidate_verb.lower().endswith("ing"):
        return False
    
    return cand_first_token.tag_ == target_tag